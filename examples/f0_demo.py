"""F0 RMSE Calculation Demo - TTS 基频评测示例"""

import numpy as np
import tempfile
from pathlib import Path


def generate_test_audio(filepath: str, fs: int = 16000, duration: float = 1.0, f0: float = 200.0, noise_level: float = 0.0):
    """生成带有特定基频的测试音频"""
    import soundfile as sf

    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    # 生成谐波信号（模拟有声语音）
    signal = np.sin(2 * np.pi * f0 * t)
    signal += 0.5 * np.sin(2 * np.pi * 2 * f0 * t)  # 二次谐波
    signal += 0.3 * np.sin(2 * np.pi * 3 * f0 * t)  # 三次谐波
    if noise_level > 0:
        signal += noise_level * np.random.randn(len(signal))
    signal = (signal * 32767 * 0.5).astype(np.int16)
    sf.write(filepath, signal, fs)


def demo_single_pair():
    """示例1: 单对音频 log-F0 RMSE 计算"""
    print("=" * 60)
    print("示例1: 单对音频 log-F0 RMSE 计算")
    print("=" * 60)

    from speech_eval.tts import compute_f0_rmse_pair

    with tempfile.TemporaryDirectory() as tmpdir:
        gt_path = Path(tmpdir) / "gt.wav"
        gen_path = Path(tmpdir) / "gen.wav"

        # ground truth: 200Hz 基频
        generate_test_audio(str(gt_path), f0=200.0)
        # generated: 略有偏移的基频（模拟合成语音音高偏差）
        generate_test_audio(str(gen_path), f0=205.0, noise_level=0.02)

        rmse = compute_f0_rmse_pair(gen_path, gt_path)
        print(f"Ground truth F0: ~200Hz")
        print(f"Generated F0:    ~205Hz")
        print(f"Log-F0 RMSE: {rmse:.4f}")
        print()


def demo_class_basic():
    """示例2: 使用 F0RMSE 类"""
    print("=" * 60)
    print("示例2: F0RMSE 类 - 基础用法")
    print("=" * 60)

    from speech_eval.tts import F0RMSE

    metric = F0RMSE(f0min=40, f0max=800)

    with tempfile.TemporaryDirectory() as tmpdir:
        gt_path = Path(tmpdir) / "gt.wav"
        gen_path = Path(tmpdir) / "gen.wav"

        generate_test_audio(str(gt_path), f0=150.0, duration=0.8)
        generate_test_audio(str(gen_path), f0=160.0, duration=0.8)

        result = metric.compute(gen_path, gt_path)
        print(f"Log-F0 RMSE: {result['log_f0_rmse']}")
        print()


def demo_batch():
    """示例3: 批量计算"""
    print("=" * 60)
    print("示例3: 批量计算多对音频")
    print("=" * 60)

    from speech_eval.tts import F0RMSE

    metric = F0RMSE()

    with tempfile.TemporaryDirectory() as tmpdir:
        gen_paths = []
        gt_paths = []

        # 不同程度的音高偏差
        f0_offsets = [2.0, 10.0, 20.0]
        for i, offset in enumerate(f0_offsets):
            gt_path = Path(tmpdir) / f"gt_{i}.wav"
            gen_path = Path(tmpdir) / f"gen_{i}.wav"
            generate_test_audio(str(gt_path), f0=200.0, duration=0.8)
            generate_test_audio(str(gen_path), f0=200.0 + offset, duration=0.8)
            gen_paths.append(gen_path)
            gt_paths.append(gt_path)

        result = metric.compute_batch(gen_paths, gt_paths)
        print(f"Mean Log-F0 RMSE: {result['mean_log_f0_rmse']}")
        print(f"Std Log-F0 RMSE:  {result['std_log_f0_rmse']}")
        print(f"Per-file details:")
        for name, val in result["details"].items():
            print(f"  {name}: {val}")
        print()


def demo_directory():
    """示例4: 基于目录的评测"""
    print("=" * 60)
    print("示例4: 基于目录的批量评测")
    print("=" * 60)

    from speech_eval.tts import F0RMSE

    metric = F0RMSE()

    with tempfile.TemporaryDirectory() as tmpdir:
        gt_dir = Path(tmpdir) / "ground_truth"
        gen_dir = Path(tmpdir) / "generated"
        output_dir = Path(tmpdir) / "results"
        gt_dir.mkdir()
        gen_dir.mkdir()

        for name in ["utt001", "utt002", "utt003"]:
            base_f0 = np.random.uniform(150, 250)
            generate_test_audio(str(gt_dir / f"{name}.wav"), f0=base_f0, duration=0.8)
            offset = np.random.uniform(5, 15)
            generate_test_audio(str(gen_dir / f"{name}.wav"), f0=base_f0 + offset, duration=0.8)

        result = metric.compute_from_dir(gen_dir, gt_dir, output_dir=output_dir)
        print(f"Mean Log-F0 RMSE: {result['mean_log_f0_rmse']}")
        print(f"Std Log-F0 RMSE:  {result['std_log_f0_rmse']}")
        print(f"Results saved to: {result['output_dir']}")
        print(f"\nutt2log_f0_rmse content:")
        print((Path(result["output_dir"]) / "utt2log_f0_rmse").read_text())


if __name__ == "__main__":
    demo_single_pair()
    demo_class_basic()
    demo_batch()
    demo_directory()
