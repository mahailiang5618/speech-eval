"""MCD Calculation Demo - TTS 梅尔倒谱距离评测示例"""

import numpy as np
import tempfile
from pathlib import Path


def generate_test_audio(filepath: str, fs: int = 16000, duration: float = 1.0, freq: float = 440.0, noise_level: float = 0.0):
    """生成测试用音频文件"""
    import soundfile as sf

    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)
    if noise_level > 0:
        signal += noise_level * np.random.randn(len(signal))
    signal = (signal * 32767 * 0.8).astype(np.int16)
    sf.write(filepath, signal, fs)


def demo_single_pair():
    """示例1: 单对音频 MCD 计算"""
    print("=" * 60)
    print("示例1: 单对音频文件 MCD 计算")
    print("=" * 60)

    from speech_eval.tts import compute_mcd_pair

    with tempfile.TemporaryDirectory() as tmpdir:
        gt_path = Path(tmpdir) / "gt.wav"
        gen_path = Path(tmpdir) / "gen.wav"

        # ground truth: 纯净 440Hz 正弦波
        generate_test_audio(str(gt_path), freq=440.0)
        # generated: 加噪版本（模拟合成语音与真实语音的差异）
        generate_test_audio(str(gen_path), freq=440.0, noise_level=0.05)

        mcd = compute_mcd_pair(gen_path, gt_path)
        print(f"Ground truth: {gt_path.name}")
        print(f"Generated:    {gen_path.name}")
        print(f"MCD: {mcd:.4f} dB")
        print()


def demo_class_basic():
    """示例2: 使用 MCD 类"""
    print("=" * 60)
    print("示例2: MCD 类 - 基础用法")
    print("=" * 60)

    from speech_eval.tts import MCD

    metric = MCD(n_fft=512, n_shift=128)

    with tempfile.TemporaryDirectory() as tmpdir:
        gt_path = Path(tmpdir) / "gt.wav"
        gen_path = Path(tmpdir) / "gen.wav"

        generate_test_audio(str(gt_path), freq=440.0, duration=0.5)
        generate_test_audio(str(gen_path), freq=445.0, duration=0.5)  # 略有频率偏差

        result = metric.compute(gen_path, gt_path)
        print(f"MCD: {result['mcd']} dB")
        print()


def demo_batch():
    """示例3: 批量计算"""
    print("=" * 60)
    print("示例3: 批量计算多对音频")
    print("=" * 60)

    from speech_eval.tts import MCD

    metric = MCD(n_fft=512, n_shift=128)

    with tempfile.TemporaryDirectory() as tmpdir:
        gen_paths = []
        gt_paths = []

        for i, noise in enumerate([0.01, 0.05, 0.1]):
            gt_path = Path(tmpdir) / f"gt_{i}.wav"
            gen_path = Path(tmpdir) / f"gen_{i}.wav"
            generate_test_audio(str(gt_path), freq=440.0, duration=0.5)
            generate_test_audio(str(gen_path), freq=440.0, duration=0.5, noise_level=noise)
            gen_paths.append(gen_path)
            gt_paths.append(gt_path)

        result = metric.compute_batch(gen_paths, gt_paths)
        print(f"Mean MCD: {result['mean_mcd']} dB")
        print(f"Std MCD:  {result['std_mcd']} dB")
        print(f"Utterances: {result['num_utterances']}")
        print(f"Per-file details: {result['details']}")
        print()


def demo_directory():
    """示例4: 基于目录的评测"""
    print("=" * 60)
    print("示例4: 基于目录的批量评测")
    print("=" * 60)

    from speech_eval.tts import MCD

    metric = MCD(n_fft=512, n_shift=128)

    with tempfile.TemporaryDirectory() as tmpdir:
        gt_dir = Path(tmpdir) / "ground_truth"
        gen_dir = Path(tmpdir) / "generated"
        output_dir = Path(tmpdir) / "results"
        gt_dir.mkdir()
        gen_dir.mkdir()

        # 创建匹配的音频文件对（按文件名匹配）
        for name in ["utt001", "utt002", "utt003"]:
            generate_test_audio(str(gt_dir / f"{name}.wav"), freq=440.0, duration=0.5)
            noise = np.random.uniform(0.02, 0.1)
            generate_test_audio(str(gen_dir / f"{name}.wav"), freq=440.0, duration=0.5, noise_level=noise)

        result = metric.compute_from_dir(gen_dir, gt_dir, output_dir=output_dir)
        print(f"Mean MCD: {result['mean_mcd']} dB")
        print(f"Std MCD:  {result['std_mcd']} dB")
        print(f"Results saved to: {result['output_dir']}")
        print(f"\nutt2mcd content:")
        print((Path(result["output_dir"]) / "utt2mcd").read_text())


if __name__ == "__main__":
    demo_single_pair()
    demo_class_basic()
    demo_batch()
    demo_directory()
