"""ASR 典型用例 - 使用 WER 方式三（详细结果 + 对齐可视化）"""

from speech_eval.asr import WER

metric = WER()


def case_1_noise_environment():
    """用例1: 嘈杂环境下的语音识别"""
    print("=" * 60)
    print("用例1: 嘈杂环境（餐厅/街道背景噪音）")
    print("=" * 60)

    refs = [
        "please reserve a table for two tonight",
        "can you turn down the music a little bit",
        "i would like to order a cup of coffee",
        "the meeting is scheduled for three o clock",
    ]
    hyps = [
        "please reserve a table for tonight",
        "can you turn down the a little bit",
        "i would like to a cup of coffee",
        "the meeting is schedule for three a clock",
    ]

    result = metric.compute_detail(refs, hyps)
    print(f"Overall WER: {result['wer']}%")
    print(f"Per-sentence WER: {result['sentence_wers']}")
    print(f"\n{result['alignment']}")


def case_2_accent_variation():
    """用例2: 口音/方言变体"""
    print("=" * 60)
    print("用例2: 口音/方言变体（非母语说话者）")
    print("=" * 60)

    refs = [
        "i think this is a very interesting problem",
        "we should discuss the budget for next quarter",
        "the data shows a significant improvement",
        "can we schedule a follow up meeting tomorrow",
    ]
    hyps = [
        "i think this is very interesting problem",
        "we should discuss the budget for next water",
        "the data shows a significant in provement",
        "can we schedule a follow up meeting to morrow",
    ]

    result = metric.compute_detail(refs, hyps)
    print(f"Overall WER: {result['wer']}%")
    print(f"Per-sentence WER: {result['sentence_wers']}")
    print(f"\n{result['alignment']}")


def case_3_medical_domain():
    """用例3: 医疗领域专业术语"""
    print("=" * 60)
    print("用例3: 医疗领域（专业术语识别挑战）")
    print("=" * 60)

    refs = [
        "the patient was diagnosed with hypertension",
        "administer ten milligrams of morphine intravenously",
        "the electrocardiogram shows atrial fibrillation",
        "schedule a magnetic resonance imaging scan",
    ]
    hyps = [
        "the patient was diagnosed with hyper tension",
        "administer ten milligrams of morphine intra vainously",
        "the electro cardio gram shows a trail fibrillation",
        "schedule a magnetic resonance image in scan",
    ]

    result = metric.compute_detail(refs, hyps)
    print(f"Overall WER: {result['wer']}%")
    print(f"Per-sentence WER: {result['sentence_wers']}")
    print(f"\n{result['alignment']}")


def case_4_voice_assistant():
    """用例4: 智能语音助手指令"""
    print("=" * 60)
    print("用例4: 智能语音助手（短指令场景）")
    print("=" * 60)

    refs = [
        "set an alarm for seven thirty tomorrow morning",
        "play my favorite playlist on spotify",
        "what is the weather forecast for this weekend",
        "send a message to mom saying i will be late",
        "navigate to the nearest gas station",
    ]
    hyps = [
        "set an alarm for seven thirty tomorrow morning",
        "play my favorite play list on spotify",
        "what is the weather forecast for this week end",
        "send a message to mom saying i will be late",
        "navigate to the nearest gas station",
    ]

    result = metric.compute_detail(refs, hyps)
    print(f"Overall WER: {result['wer']}%")
    print(f"Per-sentence WER: {result['sentence_wers']}")
    print(f"\n{result['alignment']}")


def case_5_meeting_transcription():
    """用例5: 会议转录（多人对话/长句）"""
    print("=" * 60)
    print("用例5: 会议转录（长句 + 多人对话场景）")
    print("=" * 60)

    refs = [
        "let me share my screen and walk you through the quarterly results",
        "i agree with the proposal but we need to consider the timeline",
        "the engineering team has completed the prototype ahead of schedule",
        "we need to allocate more resources to the customer support department",
    ]
    hyps = [
        "let me share my screen and walk you through the quarterly results",
        "i agree with the proposal but we need to consider the time line",
        "the engineering team has completed the proto type ahead of schedule",
        "we need to allocate more resources to the custom support department",
    ]

    result = metric.compute_detail(refs, hyps)
    print(f"Overall WER: {result['wer']}%")
    print(f"Per-sentence WER: {result['sentence_wers']}")
    print(f"\n{result['alignment']}")


if __name__ == "__main__":
    case_1_noise_environment()
    case_2_accent_variation()
    case_3_medical_domain()
    case_4_voice_assistant()
    case_5_meeting_transcription()
