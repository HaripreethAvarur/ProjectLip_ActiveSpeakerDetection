import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent / "TalkNet-ASD"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pretrain_TalkSet.model")
    parser.add_argument("--results", default="./results")
    parser.add_argument("--out", default="./checkpoints")
    parser.add_argument("--rpi-mode", action="store_true",
                        help="Restrict to 4 CPU threads to simulate RPi 4")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    return parser.parse_args()


class TalkNetVisualEncoder(nn.Module):
    """
    Wraps TalkNet's three visual sub-modules into a single benchmarkable unit.
    Input:  (B, T, H, W)
    Output: (B, 128, T)
    """
    def __init__(self, talknet_model):
        super().__init__()
        self.frontend = talknet_model.model.visualFrontend
        self.tcn      = talknet_model.model.visualTCN
        self.conv1d   = talknet_model.model.visualConv1D

    def forward(self, x):
        batch_size, num_frames, H, W = x.shape
        x = x.unsqueeze(2)
        x = x.permute(1, 0, 2, 3, 4)
        x = self.frontend(x)
        x = x.permute(1, 2, 0)
        x = self.tcn(x)
        x = self.conv1d(x)
        return x


def get_model_size_mb(model):
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / 1e6


def benchmark_model(model, input_tensor, warmup_iters, timed_iters):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_iters):
            model(input_tensor)

    latencies = []
    with torch.no_grad():
        for _ in range(timed_iters):
            t_start = time.perf_counter()
            model(input_tensor)
            latencies.append((time.perf_counter() - t_start) * 1000)

    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "fps": float(1000.0 / np.mean(latencies)),
    }


def main():
    args = parse_args()
    Path(args.results).mkdir(parents=True, exist_ok=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    if args.rpi_mode:
        torch.set_num_threads(4)
        print("[Bench] RPi simulation: 4 CPU threads")

    print("[1] Loading TalkNet on CPU...")
    from talkNet import talkNet
    talknet = talkNet()
    talknet.loadParameters(args.model)
    talknet = talknet.cpu().eval()

    visual_encoder = TalkNetVisualEncoder(talknet).cpu().eval()
    fp32_size_mb = get_model_size_mb(visual_encoder)
    print(f"FP32 size: {fp32_size_mb:.1f} MB")

    dummy_input = torch.randn(1, 11, 112, 112)
    print("[*] Verifying forward pass...")
    with torch.no_grad():
        output = visual_encoder(dummy_input)
    print(f"{tuple(dummy_input.shape)} → {tuple(output.shape)}  ✓")

    print("[2] Applying INT8 dynamic quantization...")
    int8_encoder = torch.quantization.quantize_dynamic(
        visual_encoder, {nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d}, dtype=torch.qint8)
    int8_size_mb = get_model_size_mb(int8_encoder)
    print(f"INT8 size: {int8_size_mb:.1f} MB  "
          f"({fp32_size_mb/max(int8_size_mb, 1e-6):.1f}x smaller)")

    print(f"[3] Benchmarking ({args.warmup} warmup + {args.iters} timed iterations)...")
    fp32_stats = benchmark_model(visual_encoder, dummy_input, args.warmup, args.iters)
    int8_stats = benchmark_model(int8_encoder,   dummy_input, args.warmup, args.iters)
    speedup = fp32_stats["mean_ms"] / max(int8_stats["mean_ms"], 1e-6)
    print(f"FP32: {fp32_stats['mean_ms']:.1f}ms  {fp32_stats['fps']:.1f}fps")
    print(f"INT8: {int8_stats['mean_ms']:.1f}ms  {int8_stats['fps']:.1f}fps  ({speedup:.1f}x faster)")

    print("[4] Exporting TorchScript models...")
    for encoder, name in [(visual_encoder, "fp32"), (int8_encoder, "int8")]:
        try:
            scripted = torch.jit.trace(encoder.cpu(), dummy_input)
            save_path = f"{args.out}/talknet_visual_{name}.pt"
            scripted.save(save_path)
            print(f"Saved: {save_path}")
        except Exception as e:
            print(f"{name} export warning: {e}")

    size_ratio = fp32_size_mb / max(int8_size_mb, 1e-6)
    frame_budget_pct = int8_stats["mean_ms"] / 33.3 * 100

    report = f"""
{'='*58}
  Edge Deployment Benchmark — TalkNet Visual Encoder
  INT8 Dynamic Quantization  (CPU / RPi 4 simulation)
{'='*58}

  Threads : {'4 (RPi simulation)' if args.rpi_mode else torch.get_num_threads()}
  Input   : (1, 11, 112, 112)   batch=1, T=11 frames
  Modules : visualFrontend + visualTCN + visualConv1D

  {'Metric':<26} {'FP32':>10} {'INT8':>10} {'Ratio':>8}
  {'-'*56}
  {'Model size (MB)':<26} {fp32_size_mb:>10.1f} {int8_size_mb:>10.1f} {size_ratio:>7.1f}x
  {'Mean latency (ms)':<26} {fp32_stats['mean_ms']:>10.1f} {int8_stats['mean_ms']:>10.1f} {speedup:>7.1f}x
  {'Throughput (FPS)':<26} {fp32_stats['fps']:>10.1f} {int8_stats['fps']:>10.1f} {speedup:>7.1f}x
  {'Std dev (ms)':<26} {fp32_stats['std_ms']:>10.2f} {int8_stats['std_ms']:>10.2f}      —

  INT8 is {speedup:.1f}x faster and {size_ratio:.1f}x smaller than FP32.
  At {int8_stats['fps']:.0f} FPS, the visual encoder uses {frame_budget_pct:.0f}%
  of a 30-FPS frame budget on RPi 4.

  Note: Conv3D layers (visualFrontend) dominate compute cost
  and are not quantized by dynamic INT8, which explains the
  modest speedup. Static quantization with calibration data
  would yield larger gains on the 3D convolution backbone.
{'='*58}
"""

    print(report)
    report_path = f"{args.results}/quantization_report.txt"
    Path(report_path).write_text(report)
    print(f"Saved: {report_path}")

    with open(f"{args.results}/benchmark.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "fp32", "int8"])
        writer.writerow(["size_mb", round(fp32_size_mb, 2), round(int8_size_mb, 2)])
        writer.writerow(["mean_ms", round(fp32_stats["mean_ms"],2), round(int8_stats["mean_ms"],2)])
        writer.writerow(["fps", round(fp32_stats["fps"], 1), round(int8_stats["fps"], 1)])
    print(f"Saved: {args.results}/benchmark.csv")


if __name__ == "__main__":
    main()