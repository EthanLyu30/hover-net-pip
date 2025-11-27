#!/usr/bin/env python3
"""环境 smoke test：验证 hover_net 包能在当前 Python 环境被正确调用。

脚本内容参考了 README 中的 Python API 用法：
1. 导入 hover_net 并实例化 HoVerNet（mode 可选 original/fast）。
2. 在指定设备上跑一遍 dummy forward，看各分支输出是否正常。
3. 确认 `hover_net.infer` 下的 Tile / WSI 管理器可以成功导入。
"""

from __future__ import annotations

import argparse
from pprint import pformat

import torch

import hover_net
from hover_net.infer import TileInferManager, WSIInferManager


def smoke_test_model(mode: str, nr_types: int | None, device: str) -> dict[str, tuple]:
    """按 README 的示例初始化 HoVerNet，并跑一次 dummy forward。"""
    model = hover_net.HoVerNet(input_ch=3, nr_types=nr_types, mode=mode)
    model.eval()
    model.to(device)

    # original 模式输入 270x270，fast 模式输入 256x256（见 README 描述）
    patch_size = 256 if mode == "fast" else 270
    dummy = torch.randint(
        low=0,
        high=256,
        size=(1, 3, patch_size, patch_size),
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        outputs = model(dummy)

    return {branch: tuple(tensor.shape) for branch, tensor in outputs.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="HoVer-Net 环境可用性测试")
    parser.add_argument(
        "--mode",
        default="fast",
        choices=["fast", "original"],
        help="对应 README 中的 model_mode，默认为 fast。",
    )
    parser.add_argument(
        "--nr-types",
        type=int,
        default=5,
        help="核种类数量。只做分割可传 0 或留空（会禁用类型分支）。",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="运行 dummy forward 的设备。",
    )
    args = parser.parse_args()

    nr_types = None if args.nr_types <= 0 else args.nr_types
    print(f"[+] 尝试在设备 {args.device} 上初始化 HoVerNet（mode={args.mode}, nr_types={nr_types}) …")
    output_shapes = smoke_test_model(args.mode, nr_types, args.device)
    print("[+] 模型前向计算成功，各分支输出尺寸：")
    print(pformat(output_shapes))

    # 仅提示 infer manager 已可用；实际推理流程参考 README 的 Run Inference 章节
    print("[+] 成功导入 TileInferManager:", TileInferManager)
    print("[+] 成功导入 WSIInferManager:", WSIInferManager)
    print("[✓] hover_net 包在当前环境可正常调用，具体推理/训练流程请按照 README 执行。")


if __name__ == "__main__":
    main()

