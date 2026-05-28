# Speech-Eval

语音评测工具集，汇总了 ASR（自动语音识别）、TTS（语音合成）、SVS（歌声合成）等语音任务中常用的评估指标计算代码。

## 安装

```bash
pip install speech-eval

# 含 SpeechBERTScore / SpeechBLEU 支持（需要 PyTorch）
pip install speech-eval[bert]
```

## 支持的指标

| 指标 | 模块 | 说明 | 方向 |
|------|------|------|------|
| WER | asr | 词错误率 | 越低越好 |
| CER | asr | 字错误率 | 越低越好 |
| MCD | tts | 梅尔倒谱距离 | 越低越好 |
| Log-F0 RMSE | tts | 对数基频误差 | 越低越好 |
| SpeechBERTScore | tts | 语音语义相似度 | 越高越好 |
| SpeechBLEU | tts | 语音离散token BLEU分数 | 越高越好 |
| VUV Error | common | 清浊音分类错误率 | 越低越好 |
| Semitone ACC | svs | 半音准确率 | 越高越好 |

## 快速开始

```python
# ASR
from speech_eval.asr import compute_wer, compute_cer
wer = compute_wer("the cat sat on the mat", "the cat sit on a mat")
cer = compute_cer("今天天气很好", "今天气很好")

# TTS
from speech_eval.tts import compute_mcd_pair, compute_f0_rmse_pair
mcd = compute_mcd_pair("gen.wav", "gt.wav")
f0_rmse = compute_f0_rmse_pair("gen.wav", "gt.wav")

# SVS
from speech_eval.svs import compute_semitone_acc_pair
result = compute_semitone_acc_pair("gen_singing.wav", "gt_singing.wav")

# VUV Error (TTS/SVS 通用)
from speech_eval.common import compute_vuv_error_pair
result = compute_vuv_error_pair("gen.wav", "gt.wav")
```

## 项目结构

```
speech-eval/
├── pyproject.toml
├── README.md
├── LICENSE
├── speech_eval/
│   ├── __init__.py
│   ├── asr/            # WER, CER
│   ├── tts/            # MCD, F0 RMSE, SpeechBERTScore, SpeechBLEU
│   ├── svs/            # Semitone ACC
│   └── common/         # VUV Error (TTS/SVS 共享)
├── examples/           # 使用示例
└── docs/               # 各模块详细文档
```

## 文档

各模块的详细用法、API 文档和指标原理请参阅：
- [docs/asr.md](docs/asr.md) — WER、CER 详细文档
- [docs/tts.md](docs/tts.md) — MCD、F0 RMSE、SpeechBERTScore、SpeechBLEU、VUV Error 详细文档
- [docs/svs.md](docs/svs.md) — Semitone ACC、VUV Error 详细文档

## 运行示例

```bash
pip install speech-eval
python examples/wer_demo.py
python examples/cer_demo.py
python examples/mcd_demo.py
python examples/f0_demo.py
python examples/semitone_demo.py
python examples/speech_bert_score_demo.py
python examples/vuv_demo.py
```

## 参考来源

- [ESPnet](https://github.com/espnet/espnet) — 语音处理工具包
- [jiwer](https://github.com/jitsi/jiwer) — WER/CER 计算
- [pysptk](https://github.com/r9y9/pysptk) — 梅尔倒谱提取
- [pyworld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) — F0 提取
- [fastdtw](https://github.com/slaypni/fastdtw) — 动态时间规整
- [discrete-speech-metrics](https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics) — SpeechBERTScore & SpeechBLEU

## License

MIT
