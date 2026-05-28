"""Semitone ACC Calculation Demo - 歌声合成半音准确率评测示例"""

import numpy as np
import tempfile
from pathlib import Path


def generate_singing_audio(filepath: str, fs: int = 24000, duration: float = 1.0, f0: float = 440.0, vibrato: float = 0.0):
    """生成模拟歌声的测试音频（含谐波结构）"""
    import soundfile as sf

    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    # 添加vibrato（颤音）模拟真实歌声
    f0_contour = f0 + vibrato * np.sin(2 * np.pi * 5 * t)
    phase = 2 * np.pi * np.cumsum(f0_contour) / fs
    # 谐波叠加模拟歌声音色
    signal = np.sin(phase)
    signal += 0.6 * np.sin(2 * phase)
    signal += 0.4 * np.sin(3 * phase)
    signal += 0.2 * np.sin(4 * phase)
    signal = (signal * 32767 * 0.4).astype(np.int16)
    sf.write(filepath, signal, fs)


def demo_hz_to_semitone():
    """示例1: Hz 到半音音符转换"""
    print("=" * 60)
    print("示例1: Hz → Semitone 转换")
    print("=" * 60)

    from speech_eval.svs import hz_to_semitone

    test_freqs = [0, 261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
    note_labels = ["Silence", "C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]

    for freq, label in zip(test_freqs, note_labels):
        semitone = hz_to_semitone(freq)
        print(f"  {freq:>7.2f} Hz  →  {semitone:<6s}  (expected: {label})")
    print()


def demo_single_pair():
    """示例2: 单对音频 Semitone ACC 计算"""
    print("=" * 60)
    print("示例2: 单对音频 Semitone ACC")
    print("=" * 60)

    from speech_eval.svs import compute_semitone_acc_pair

    with tempfile.TemporaryDirectory() as tmpdir:
        gt_path = Path(tmpdir) / "gt.wav"
        gen_path = Path(tmpdir) / "gen.wav"

        # ground truth: A4 (440Hz)
        generate_singing_audio(str(gt_path), f0=440.0, vibrato=3.0)
        # generated: 略有音高偏差但仍在同一半音内
        generate_singing_audio(str(gen_path), f0=442.0, vibrato=3.0)

        result = compute_semitone_acc_pair(gen_path, gt_path)
        print(f"Ground truth: A4 (~440Hz)")
        print(f"Generated:    ~442Hz (same semitone)")
        print(f"Semitone ACC: {result['semitone_acc_pct']}%")
        print(f"Total frames: {result['total_frames']}")
        print()


def demo_pitch_shift():
    """示例3: 不同音高偏移的对比"""
    print("=" * 60)
    print("示例3: 不同音高偏移对 Semitone ACC 的影响")
    print("=" * 60)

    from speech_eval.svs import compute_semitone_acc_pair

    base_f0 = 440.0  # A4
    offsets = [0, 5, 15, 30, 50]  # Hz偏移

    with tempfile.TemporaryDirectory() as tmpdir:
        gt_path = Path(tmpdir) / "gt.wav"
        generate_singing_audio(str(gt_path), f0=base_f0, duration=0.8)

        for offset in offsets:
            gen_path = Path(tmpdir) / f"gen_offset_{offset}.wav"
            generate_singing_audio(str(gen_path), f0=base_f0 + offset, duration=0.8)
            result = compute_semitone_acc_pair(gen_path, gt_path)
            print(f"  Offset +{offset:>2d}Hz ({base_f0 + offset:.0f}Hz) → Semitone ACC: {result['semitone_acc_pct']}%")
    print()


def demo_class_batch():
    """示例4: 使用 SemitoneACC 类批量计算"""
    print("=" * 60)
    print("示例4: SemitoneACC 类 - 批量计算")
    print("=" * 60)

    from speech_eval.svs import SemitoneACC

    metric = SemitoneACC(f0min=40, f0max=800)

    with tempfile.TemporaryDirectory() as tmpdir:
        gen_paths = []
        gt_paths = []

        # 模拟不同音符的歌声片段
        notes = [(261.63, 263.0), (440.0, 445.0), (523.25, 530.0)]  # (gt_f0, gen_f0)
        for i, (gt_f0, gen_f0) in enumerate(notes):
            gt_path = Path(tmpdir) / f"gt_{i}.wav"
            gen_path = Path(tmpdir) / f"gen_{i}.wav"
            generate_singing_audio(str(gt_path), f0=gt_f0, duration=0.6)
            generate_singing_audio(str(gen_path), f0=gen_f0, duration=0.6)
            gen_paths.append(gen_path)
            gt_paths.append(gt_path)

        result = metric.compute_batch(gen_paths, gt_paths)
        print(f"Mean Semitone ACC: {result['mean_semitone_acc_pct']}%")
        print(f"Std:  {result['std_semitone_acc_pct']}%")
        print(f"Per-file details:")
        for name, val in result["details"].items():
            print(f"  {name}: {val}%")
        print()


def demo_directory():
    """示例5: 基于目录的评测"""
    print("=" * 60)
    print("示例5: 基于目录的批量评测")
    print("=" * 60)

    from speech_eval.svs import SemitoneACC

    metric = SemitoneACC()

    with tempfile.TemporaryDirectory() as tmpdir:
        gt_dir = Path(tmpdir) / "ground_truth"
        gen_dir = Path(tmpdir) / "generated"
        output_dir = Path(tmpdir) / "results"
        gt_dir.mkdir()
        gen_dir.mkdir()

        for name, f0 in [("song_001", 330.0), ("song_002", 440.0), ("song_003", 523.0)]:
            generate_singing_audio(str(gt_dir / f"{name}.wav"), f0=f0, duration=0.6, vibrato=4.0)
            offset = np.random.uniform(0, 20)
            generate_singing_audio(str(gen_dir / f"{name}.wav"), f0=f0 + offset, duration=0.6, vibrato=4.0)

        result = metric.compute_from_dir(gen_dir, gt_dir, output_dir=output_dir)
        print(f"Mean Semitone ACC: {result['mean_semitone_acc_pct']}%")
        print(f"Results saved to: {result['output_dir']}")
        print(f"\nutt2semitone_acc content:")
        print((Path(result["output_dir"]) / "utt2semitone_acc").read_text())


if __name__ == "__main__":
    demo_hz_to_semitone()
    demo_single_pair()
    demo_pitch_shift()
    demo_class_batch()
    demo_directory()
