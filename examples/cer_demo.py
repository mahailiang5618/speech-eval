"""CER Calculation Demo - 语音识别字错误率评测示例"""

from speech_eval.asr import CER, compute_cer


def demo_quick_function():
    """示例1: 使用快捷函数 compute_cer"""
    print("=" * 60)
    print("示例1: 快捷函数 compute_cer")
    print("=" * 60)

    # 中文 CER
    ref = "今天天气很好"
    hyp = "今天气很好"
    cer = compute_cer(ref, hyp)
    print(f"Reference:  {ref}")
    print(f"Hypothesis: {hyp}")
    print(f"CER: {cer}%")
    print()

    # 英文 CER（字符级别）
    ref_en = "hello world"
    hyp_en = "helo word"
    cer_en = compute_cer(ref_en, hyp_en)
    print(f"Reference:  {ref_en}")
    print(f"Hypothesis: {hyp_en}")
    print(f"CER: {cer_en}%")
    print()


def demo_batch():
    """示例2: 批量计算"""
    print("=" * 60)
    print("示例2: 批量计算多条数据")
    print("=" * 60)

    refs = [
        "语音识别技术",
        "自然语言处理",
        "深度学习模型",
    ]
    hyps = [
        "语音识别技",
        "自然语言处理",
        "深度学习模形",
    ]

    cer = compute_cer(refs, hyps)
    print(f"Batch CER: {cer}%")
    print()


def demo_class_basic():
    """示例3: 使用 CER 类获取详细信息"""
    print("=" * 60)
    print("示例3: CER 类 - 基础用法")
    print("=" * 60)

    metric = CER(lowercase=True, remove_punctuation=True)

    ref = "北京市海淀区中关村大街"
    hyp = "北京市海淀中关村大街"

    result = metric.compute(ref, hyp)
    print(f"Reference:  {ref}")
    print(f"Hypothesis: {hyp}")
    print(f"CER: {result['cer']}%")
    print(f"Substitutions: {result['substitutions']}")
    print(f"Deletions: {result['deletions']}")
    print(f"Insertions: {result['insertions']}")
    print(f"Reference char count: {result['ref_char_count']}")
    print()


def demo_class_detail():
    """示例4: 获取逐句 CER 和对齐可视化"""
    print("=" * 60)
    print("示例4: CER 类 - 详细结果与对齐可视化")
    print("=" * 60)

    metric = CER()

    refs = [
        "人工智能改变世界",
        "机器学习算法优化",
    ]
    hyps = [
        "人工智能改变时界",
        "机器学习算法优化",
    ]

    result = metric.compute_detail(refs, hyps)
    print(f"Overall CER: {result['cer']}%")
    print(f"Per-sentence CER: {result['sentence_cers']}")
    print(f"\nAlignment visualization:\n{result['alignment']}")


def demo_file_based():
    """示例5: 基于文件的评测"""
    print("=" * 60)
    print("示例5: 基于文件的评测")
    print("=" * 60)

    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        ref_file = Path(tmpdir) / "ref.txt"
        hyp_file = Path(tmpdir) / "hyp.txt"
        output_dir = Path(tmpdir) / "results"

        ref_file.write_text(
            "今天天气很好\n明天会下雨吗\n", encoding="utf-8"
        )
        hyp_file.write_text(
            "今天气很好\n明天会下雨吗\n", encoding="utf-8"
        )

        metric = CER()
        result = metric.compute_from_file(ref_file, hyp_file, output_dir=output_dir)

        print(f"CER: {result['cer']}%")
        print(f"Alignment saved to: {result['alignment_file']}")
        print(f"\nAlignment content:")
        print(Path(result["alignment_file"]).read_text())


if __name__ == "__main__":
    demo_quick_function()
    demo_batch()
    demo_class_basic()
    demo_class_detail()
    demo_file_based()
