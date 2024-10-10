import torch
import torch.nn as nn

from lib.models.nn import (
    ConvLayer,
    DSConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResidualBlock,
)


__all__ = []


class MyNetBackbone(nn.Module):
    def __init__(
        self,
        width_list: list[int],
        depth_list: list[int],
        in_channels=3,
        expand_ratio=4,
        norm="bn2d",
        act_func="relu6",
    ):
        super(MyNetBackbone, self).__init__()

        self.width_list = []
        self.input_stem = [
            ConvLayer(
                in_channels=in_channels,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
            )
            self.input_stem.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.input_stem = OpSequential(self.input_stem)
        self.width_list.append(width_list[0])

        self.stages = []
        for w, d in zip(width_list[1:], depth_list[1:]):
            stage = []
            for i in range(d):
                stride = 2 if i == 0 else 1
                block = self.build_local_block(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                )
                stage.append(
                    ResidualBlock(block, IdentityLayer() if stride == 1 else None)
                )
                in_channels = w
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        self.stages = nn.ModuleList(self.stages)

    @staticmethod
    def build_local_block(
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: float = 4,
        norm="bn2d",
        act_func="relu6",
    ):
        if expand_ratio == 1:
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False),
                norm=(None, norm),
                act_func=(act_func, None),
            )
        else:
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False),
                norm=(None, None, norm),
                act_func=(act_func, act_func, None),
            )
        return block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_dict = {"input": x}
        output_dict["stage0"] = x = self.input_stem(x)
        for stage_id, stage in enumerate(self.stages, 1):
            output_dict[f"stage{stage_id}"] = x = stage(x)
        output_dict["stage_final"] = x
        return output_dict


def mynet_backbone_b0(**kwargs) -> MyNetBackbone:
    backbone = MyNetBackbone(
        width_list=[8, 16, 32, 64, 128], depth_list=[1, 2, 2, 2, 2], dim=16, **kwargs
    )

    return backbone


def mynet_backbone_b1(**kwargs) -> MyNetBackbone:
    backbone = MyNetBackbone(
        width_list=[16, 32, 64, 128, 256], depth_list=[1, 2, 3, 3, 4], dim=16, **kwargs
    )

    return backbone


def mynet_backbone_b2(**kwargs) -> MyNetBackbone:
    backbone = MyNetBackbone(
        width_list=[24, 48, 96, 192, 384], depth_list=[1, 3, 4, 4, 6], dim=32, **kwargs
    )

    return backbone


def mynet_backbone_b3(**kwargs) -> MyNetBackbone:
    backbone = MyNetBackbone(
        width_list=[32, 64, 128, 256, 512], depth_list=[1, 4, 6, 6, 9], dim=32, **kwargs
    )

    return backbone


if __name__ == "__main__":
    b0_backbone = mynet_backbone_b0()

    sample_input = torch.randn(1, 3, 224, 224)
    output_dict = b0_backbone(sample_input)

    for key, value in output_dict.items():
        print(f"{key}: {value.shape}")
