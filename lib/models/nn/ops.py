from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from lib.models.nn.act import build_act
from lib.models.nn.norm import build_norm
from lib.models.utils import (
    get_same_padding,
    list_sum,
    resize,
    val2list,
    val2tuple,
)

__all__ = [
    "ConvLayer",
    "UpSampleLayer",
    "LinearLayer",
    "IdentityLayer",
    "DSConv",
    "MBConv",
    "FusedMBConv",
    "ResBlock",
    "LiteMLA",
    "EfficientViTBlock",
    "ResidualBlock",
    "DAGBlock",
    "OpSequential",
]


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        use_bias: bool = False,
        dropout: float = 0,
        norm: str = "bn2d",
        act_func: str = "relu6",
    ) -> None:
        super(ConvLayer, self).__init__()

        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout: Optional[nn.Module] = (
            nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        )
        self.conv: nn.Conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm: Optional[nn.Module] = build_norm(norm, num_features=out_channels)
        self.act: Optional[nn.Module] = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class UpSampleLayer(nn.Module):
    def __init__(
        self,
        mode: str = "bicubic",
        size: tuple[int, int] | None = None,
        factor: int = 2,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    @autocast(device_type="cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.size is not None and tuple(x.shape[-2:]) == self.size
        ) or self.factor == 1:
            return x
        if x.dtype in [torch.float16, torch.bfloat16]:
            x = x.float()
        return resize(
            x,
            size=self.size,
            scale_factor=self.factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )


class LinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        dropout: float = 0,
        norm: str | None = None,
        act_func: str | None = None,
    ) -> None:
        super().__init__()
        self.dropout: nn.Dropout | None = (
            nn.Dropout(dropout, inplace=False) if dropout > 0 else None
        )
        self.linear: nn.Linear = nn.Linear(in_features, out_features, bias=use_bias)
        self.norm: nn.Module | None = build_norm(norm, num_features=out_features)
        self.act: nn.Module | None = build_act(act_func)

    def _try_squeeze(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._try_squeeze(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_bias: bool = False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super(DSConv, self).__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.depthwise_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.pointwise_conv = ConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6.0,
        use_bias=False,
        norm=("bn2d", "bn2d", "bn2d"),
        act_func=("relu6", "relu6", None),
    ):
        super(MBConv, self).__init__()

        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)

        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.depthwise_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )
        self.pointwise_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class FusedMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6.0,
        groups=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super(FusedMBConv, self).__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.spatial_conv = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            groups=groups,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.pointwise_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(x)
        x = self.pointwise_conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=1.0,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super(ResBlock, self).__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.conv1 = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.conv2 = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LiteMLA(nn.Module):
    """Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int | None = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm=("None", "bn2d"),
        act_func=(None, None),
        kernel_func="relu6",
        scales: tuple[int, ...] = (5,),
        eps=1.0e-15,
    ):
        super(LiteMLA, self).__init__()

        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        1,
                        groups=3 * heads,
                        bias=use_bias[0],
                    ),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )

    @autocast(device_type="cuda", enabled=False)
    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        if qkv.dtype == torch.float16:
            qkv = qkv.float()

        qkv = torch.reshape(qkv, (B, -1, 3 * self.dim, H * W))
        q, k, v = torch.split(qkv, self.dim, dim=2)

        # lightweight linear attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # linear matmul
        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
        vk = torch.matmul(v, trans_k)
        out = torch.matmul(vk, q)
        if out.dtype == torch.bfloat16:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

        out = torch.reshape(out, (B, -1, H, W))

        return out

    @autocast(device_type="cuda", enabled=False)
    def relu_quadratic_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        qkv = torch.reshape(qkv, (B, -1, 3 * self.dim, H * W))
        q, k, v = torch.split(qkv, self.dim, dim=2)

        q = self.kernel_func(q)
        k = self.kernel_func(k)

        att_map = torch.matmul(k.transpose(-1, -2), q)
        original_dtype = att_map.dtype
        if original_dtype in [torch.float16, torch.bfloat16]:
            att_map = att_map.float()
        att_map = att_map / (torch.sum(att_map, dim=2, keepdim=True) + self.eps)
        att_map = att_map.to(original_dtype)
        out = torch.matmul(v, att_map)

        out = torch.reshape(out, (B, -1, H, W))

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        qkv = torch.cat(multi_scale_qkv, dim=1)

        H, W = list(qkv.size()[-2:])
        if H * W > self.dim:
            out = self.relu_linear_att(qkv)
        else:
            out = self.relu_quadratic_att(qkv)
        out = self.proj(out)

        return out


class EfficientViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        heads_ratio: float = 1.0,
        dim=32,
        expand_ratio: float = 4,
        scales=(5,),
        norm="bn2d",
        act_func="relu6",
    ):
        super(EfficientViTBlock, self).__init__()

        self.context_module = ResidualBlock(
            LiteMLA(
                in_channels,
                in_channels,
                heads_ratio=heads_ratio,
                dim=dim,
                norm=(None, norm),
                scales=scales,
            ),
            IdentityLayer(),
        )
        local_module = MBConv(
            in_channels,
            in_channels,
            expand_ratio=expand_ratio,
            use_bias=(True, True, False),
            norm=(None, None, norm),
            act_func=(act_func, act_func, None),
        )
        self.local_module = ResidualBlock(local_module, IdentityLayer())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.local_module(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: nn.Module | None,
        shortcut: nn.Module | None,
        post_act=None,
        pre_norm: nn.Module | None = None,
    ):
        super(ResidualBlock, self).__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act is not None:
                res = self.post_act(res)
        return res


class DAGBlock(nn.Module):
    def __init__(
        self,
        inputs: dict[str, nn.Module],
        merge: str,
        post_input: nn.Module | None,
        middle: nn.Module,
        outputs: dict[str, nn.Module],
    ):
        super(DAGBlock, self).__init__()

        self.input_keys = list(inputs.keys())
        self.input_ops = nn.ModuleList(list(inputs.values()))
        self.merge = merge
        self.post_input = post_input

        self.middle = middle

        self.output_keys = list(outputs.keys())
        self.output_ops = nn.ModuleList(list(outputs.values()))

    def forward(self, feature_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feat = [
            op(feature_dict[key]) for key, op in zip(self.input_keys, self.input_ops)
        ]
        if self.merge == "add":
            feat = list_sum(feat)
        elif self.merge == "cat":
            feat = torch.cat(feat, dim=1)
        else:
            raise NotImplementedError
        if self.post_input is not None:
            feat = self.post_input(feat)
        feat = self.middle(feat)
        for key, op in zip(self.output_keys, self.output_ops):
            feature_dict[key] = op(feat)
        return feature_dict


class OpSequential(nn.Module):
    def __init__(self, op_list: list[nn.Module | None]):
        super(OpSequential, self).__init__()

        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)

        self.op_list = nn.ModuleList(valid_op_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x
