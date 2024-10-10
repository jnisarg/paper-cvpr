import torch
import torch.nn as nn

from lib.models.mynet.backbone import MyNetBackbone
from lib.models.nn import (
    ConvLayer,
    DAGBlock,
    FusedMBConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResidualBlock,
    UpSampleLayer,
)


__all__ = ["MyNetSeg", "mynet_seg_b0", "mynet_seg_b1", "mynet_seg_b2", "mynet_seg_b3"]


class SegHead(DAGBlock):
    def __init__(
        self,
        fid_list: list[str],
        in_channel_list: list[int],
        stride_list: list[int],
        head_stride: int,
        head_width: int,
        head_depth: int,
        expand_ratio: float,
        middle_op: str,
        final_expand: float | None,
        n_classes: int,
        dropout=0,
        norm="bn2d",
        act_func="relu6",
    ):
        inputs = {}
        for fid, in_channel, stride in zip(fid_list, in_channel_list, stride_list):
            factor = stride // head_stride
            if factor == 1:
                inputs[fid] = ConvLayer(
                    in_channel, head_width, 1, norm=norm, act_func=None
                )
            else:
                inputs[fid] = OpSequential(
                    [
                        ConvLayer(in_channel, head_width, 1, norm=norm, act_func=None),
                        UpSampleLayer(factor=factor),
                    ]
                )

        middle = []
        for _ in range(head_depth):
            if middle_op == "mbconv":
                block = MBConv(
                    in_channels=head_width,
                    out_channels=head_width,
                    stride=1,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, act_func, None),
                )
            elif middle_op == "fmbconv":
                block = FusedMBConv(
                    in_channels=head_width,
                    out_channels=head_width,
                    stride=1,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, None),
                )
            else:
                raise NotImplementedError
            middle.append(ResidualBlock(block, IdentityLayer()))
        middle = OpSequential(middle)

        outputs = {
            "segout": OpSequential(
                [
                    (
                        None
                        if final_expand is None
                        else ConvLayer(
                            head_width,
                            int(head_width * final_expand),
                            1,
                            norm=norm,
                            act_func=act_func,
                        )
                    ),
                    ConvLayer(
                        int(head_width * (final_expand or 1)),
                        n_classes,
                        1,
                        use_bias=True,
                        dropout=dropout,
                        norm=None,
                        act_func=None,
                    ),
                    UpSampleLayer(factor=head_stride),
                ]
            )
        }

        super(SegHead, self).__init__(inputs, "add", None, middle, outputs)


class MyNetSeg(nn.Module):
    def __init__(self, backbone: MyNetBackbone, head: SegHead) -> None:
        super(MyNetSeg, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feed_dict = self.backbone(x)
        feed_dict = self.head(feed_dict)
        return feed_dict["segout"]


def mynet_seg_b0(n_classes: int, **kwargs) -> MyNetSeg:
    from lib.models.mynet.backbone import mynet_backbone_b0

    backbone = mynet_backbone_b0(**kwargs)

    head = SegHead(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[128, 64, 32],
        stride_list=[32, 16, 8],
        head_stride=8,
        head_width=32,
        head_depth=1,
        expand_ratio=4.0,
        middle_op="mbconv",
        final_expand=4.0,
        n_classes=n_classes,
        **kwargs,
    )

    model = MyNetSeg(backbone, head)

    return model


def mynet_seg_b1(n_classes: int, **kwargs) -> MyNetSeg:
    from lib.models.mynet.backbone import mynet_backbone_b1

    backbone = mynet_backbone_b1(**kwargs)

    head = SegHead(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[256, 128, 64],
        stride_list=[32, 16, 8],
        head_stride=8,
        head_width=64,
        head_depth=3,
        expand_ratio=4.0,
        middle_op="mbconv",
        final_expand=4.0,
        n_classes=n_classes,
        **kwargs,
    )

    model = MyNetSeg(backbone, head)

    return model


def mynet_seg_b2(n_classes: int, **kwargs) -> MyNetSeg:
    from lib.models.mynet.backbone import mynet_backbone_b2

    backbone = mynet_backbone_b2(**kwargs)

    head = SegHead(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[384, 192, 96],
        stride_list=[32, 16, 8],
        head_stride=8,
        head_width=96,
        head_depth=3,
        expand_ratio=4.0,
        middle_op="mbconv",
        final_expand=4.0,
        n_classes=n_classes,
        **kwargs,
    )

    model = MyNetSeg(backbone, head)

    return model


def mynet_seg_b3(n_classes: int, **kwargs) -> MyNetSeg:
    from lib.models.mynet.backbone import mynet_backbone_b3

    backbone = mynet_backbone_b3(**kwargs)

    head = SegHead(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        stride_list=[32, 16, 8],
        head_stride=8,
        head_width=128,
        head_depth=3,
        expand_ratio=4.0,
        middle_op="mbconv",
        final_expand=4.0,
        n_classes=n_classes,
        **kwargs,
    )

    model = MyNetSeg(backbone, head)

    return model


if __name__ == "__main__":
    b1_seg = mynet_seg_b1(n_classes=19)

    sample_input = torch.randn(1, 3, 1024, 2048)
    output = b1_seg(sample_input)

    print(output.shape)

    parameters = sum(p.numel() for p in b1_seg.parameters() if p.requires_grad)
    print(parameters / 1e6)
