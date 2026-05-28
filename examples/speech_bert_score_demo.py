"""SpeechBERTScore Calculation Demo - TTS 语音质量评测示例

注意: 此 demo 需要安装 discrete-speech-metrics 包和 PyTorch，
首次运行会自动下载 WavLM-Large 模型（约 1.2GB）。

安装依赖:
    pip install discrete-speech-metrics torch
"""

import numpy as np
import tempfile
from pathlib import Path


def generate_test_audio(filepath: str, fs: int = 16000, duration: float = 1.5, freq: float = 200.0, noise_level: float = 0.0):
    """生成测试音频"""
    import soundfile as sf

    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)
    signal += 0.5 * np.sin(2 * np.pi * 2 * freq * t)
    signal += 0.3 * np.sin(2 * np.pi * 3 * freq * t)
    if noise_level > 0:
        signal += noise_level * np.random.randn(len(signal))
    signal = (signal * 32767 * 0.5).astype(np.int16)
    sf.write(filepath, signal, fs)


def demo_single_pair():
    """示例1: 单对音频 SpeechBERTScore"""
    print("=" * 60)
    print("示例1: 单对音频 SpeechBERTScore")
    print("=" * 60)

    from speech_eval.tts import compute_speech_bert_score

    with tempfile.TemporaryDirectory() as tmpdir:
        gt_path = Path(tmpdir) / "gt.wav"
        gen_path = Path(tmpdir) / "gen.wav"

        generate_test_audio(str(gt_path), freq=200.0)
        generate_test_audio(str(gen_path), freq=200.0, noise_level=0.05)

        score = compute_speech_bert_score(gen_path, gt_path, use_gpu=False)
        print(f"SpeechBERTScore: {score}")
        print()


def demo_class_basic():
    """示例2: 使用 SpeechBERTScoreMetric 类"""
    print("=" * 60)
    print("示例2: SpeechBERTScoreMetric 类 - 基础用法")
    print("=" * 60)

    from speech_eval.tts import SpeechBERTScoreMetric

    metric = SpeechBERTScoreMetric(model_type="wavlm-large", layer=14, use_gpu=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        gt_path = Path(tmpdir) / "gt.wav"
        gen_path = Path(tmpdir) / "gen.wav"

        generate_test_audio(str(gt_path), freq=200.0, duration=1.0)
        generate_test_audio(str(gen_path), freq=200.0, duration=1.0, noise_level=0.02)

        result = metric.compute(gen_path, gt_path)
        print(f"SpeechBERTScore: {result['speech_bert_score']}")
        print(f"Precision: {result['precision']}")
        print(f"Recall: {result['recall']}")
        print()


def demo_batch():
    """示例3: 批量计算"""
    print("=" * 60)
    print("示例3: 批量计算多对音频")
    print("=" * 60)

    from speech_eval.tts import SpeechBERTScoreMetric

    metric = SpeechBERTScoreMetric(use_gpu=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        gen_paths = []
        gt_paths = []

        for i, noise in enumerate([0.01, 0.05, 0.2]):
            gt_path = Path(tmpdir) / f"gt_{i}.wav"
            gen_path = Path(tmpdir) / f"gen_{i}.wav"
            generate_test_audio(str(gt_path), freq=200.0, duration=1.0)
            generate_test_audio(str(gen_path), freq=200.0, duration=1.0, noise_level=noise)
            gen_paths.append(gen_path)
            gt_paths.append(gt_path)

        result = metric.compute_batch(gen_paths, gt_paths)
        print(f"Mean SpeechBERTScore: {result['mean_score']}")
        print(f"Std: {result['std_score']}")
        print(f"Per-file: {result['details']}")
        print()


def demo_directory():
    """示例4: 基于目录的评测"""
    print("=" * 60)
    print("示例4: 基于目录的批量评测")
    print("=" * 60)

    from speech_eval.tts import SpeechBERTScoreMetric

    metric = SpeechBERTScoreMetric(use_gpu=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        gt_dir = Path(tmpdir) / "ground_truth"
        gen_dir = Path(tmpdir) / "generated"
        output_dir = Path(tmpdir) / "results"
        gt_dir.mkdir()
        gen_dir.mkdir()

        for name in ["utt001", "utt002", "utt003"]:
            generate_test_audio(str(gt_dir / f"{name}.wav"), freq=200.0, duration=1.0)
            noise = np.random.uniform(0.02, 0.1)
            generate_test_audio(str(gen_dir / f"{name}.wav"), freq=200.0, duration=1.0, noise_level=noise)

        result = metric.compute_from_dir(gen_dir, gt_dir, output_dir=output_dir)
        print(f"Mean SpeechBERTScore: {result['mean_score']}")
        print(f"Std: {result['std_score']}")
        print(f"Results saved to: {result['output_dir']}")
        print(f"\nutt2spbs content:")
        print((Path(result["output_dir"]) / "utt2spbs").read_text())


if __name__ == "__main__":
    print("注意: 首次运行需下载 WavLM-Large 模型 (~1.2GB)")
    print()
    demo_single_pair()
    demo_class_basic()
    demo_batch()
    demo_directory()
