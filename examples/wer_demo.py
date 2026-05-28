"""WER Calculation Demo - 语音识别词错误率评测示例"""

from speech_eval.asr import WER, compute_wer


def demo_quick_function():
    """示例1: 使用快捷函数 compute_wer"""
    print("=" * 60)
    print("示例1: 快捷函数 compute_wer")
    print("=" * 60)

    # 英文单句
    ref = "the cat sat on the mat"
    hyp = "the cat sit on a mat"
    wer = compute_wer(ref, hyp)
    print(f"Reference:  {ref}")
    print(f"Hypothesis: {hyp}")
    print(f"WER: {wer}%")
    print()

    # 中文（按字切分，用空格分隔每个字）
    ref_cn = "今 天 天 气 很 好"
    hyp_cn = "今 天 气 很 好 啊"
    wer_cn = compute_wer(ref_cn, hyp_cn)
    print(f"Reference:  {ref_cn}")
    print(f"Hypothesis: {hyp_cn}")
    print(f"WER: {wer_cn}%")
    print()


def demo_batch():
    """示例2: 批量计算"""
    print("=" * 60)
    print("示例2: 批量计算多条数据")
    print("=" * 60)

    refs = [
        "hello world",
        "how are you doing today",
        "speech recognition is amazing",
    ]
    hyps = [
        "hello word",
        "how are you doing today",
        "speech recognition is a mazing",
    ]

    wer = compute_wer(refs, hyps)
    print(f"Batch WER: {wer}%")
    print()


def demo_class_basic():
    """示例3: 使用 WER 类获取详细信息"""
    print("=" * 60)
    print("示例3: WER 类 - 基础用法")
    print("=" * 60)

    metric = WER(lowercase=True, remove_punctuation=True)

    ref = "The quick brown fox jumps over the lazy dog"
    hyp = "the quick brown fox jump over a lazy dog"

    result = metric.compute(ref, hyp)
    print(f"Reference:  {ref}")
    print(f"Hypothesis: {hyp}")
    print(f"WER: {result['wer']}%")
    print(f"Substitutions: {result['substitutions']}")
    print(f"Deletions: {result['deletions']}")
    print(f"Insertions: {result['insertions']}")
    print(f"Reference word count: {result['ref_word_count']}")
    print()


def demo_class_detail():
    """示例4: 获取逐句 WER 和对齐可视化"""
    print("=" * 60)
    print("示例4: WER 类 - 详细结果与对齐可视化")
    print("=" * 60)

    metric = WER()

    refs = [
        "i have a dream",
        "the weather is nice today",
    ]
    hyps = [
        "i have dream",
        "the weather is nice today",
    ]

    result = metric.compute_detail(refs, hyps)
    print(f"Overall WER: {result['wer']}%")
    print(f"Per-sentence WER: {result['sentence_wers']}")
    print(f"\nAlignment visualization:\n{result['alignment']}")


def demo_file_based():
    """示例5: 基于文件的评测"""
    print("=" * 60)
    print("示例5: 基于文件的评测")
    print("=" * 60)

    from pathlib import Path
    import tempfile

    # 创建临时文件模拟真实场景
    with tempfile.TemporaryDirectory() as tmpdir:
        ref_file = Path(tmpdir) / "ref.txt"
        hyp_file = Path(tmpdir) / "hyp.txt"
        output_dir = Path(tmpdir) / "results"

        ref_file.write_text(
            "the cat sat on the mat\nhow are you doing\n", encoding="utf-8"
        )
        hyp_file.write_text(
            "the cat sit on a mat\nhow are you\n", encoding="utf-8"
        )

        metric = WER()
        result = metric.compute_from_file(ref_file, hyp_file, output_dir=output_dir)

        print(f"WER: {result['wer']}%")
        print(f"Alignment saved to: {result['alignment_file']}")
        print(f"\nAlignment content:")
        print(Path(result["alignment_file"]).read_text())


if __name__ == "__main__":
    demo_quick_function()
    demo_batch()
    demo_class_basic()
    demo_class_detail()
    demo_file_based()
