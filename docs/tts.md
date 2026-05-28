# TTS 评测指标

语音合成（TTS）评测指标实现，包含 MCD、Log-F0 RMSE、SpeechBERTScore 和 SpeechBLEU。

## 指标概览

| 指标 | 说明 | 方向 |
|------|------|------|
| MCD (Mel-Cepstral Distortion) | 梅尔倒谱距离，频谱差异 | 越低越好 |
| Log-F0 RMSE | 对数基频均方根误差，音高准确度 | 越低越好 |
| SpeechBERTScore | 基于 WavLM 的语义相似度 | 越高越好 |
| SpeechBLEU | 基于 HuBERT 离散 token 的 BLEU 分数 | 越高越好 |
| VUV Error | 清浊音帧级分类错误率 | 越低越好 |

## 安装依赖

```bash
# 基础指标 (MCD, F0 RMSE)
pip install librosa pysptk pyworld soundfile fastdtw scipy

# SpeechBERTScore & SpeechBLEU (可选，需 PyTorch)
pip install discrete-speech-metrics torch
```

## MCD (Mel-Cepstral Distortion)

### 原理

$$MCD = \frac{10}{N \cdot \ln10} \sum_{n=1}^{N} \sqrt{2 \sum_{d=1}^{D} (mc^{gen}_{n,d} - mc^{gt}_{n,d})^2}$$

- 基于 SPTK 提取梅尔倒谱系数 (MCEP)
- 通过 DTW 对齐合成与真实语音的帧序列
- 计算对齐后帧间欧氏距离，转换为 dB
- 通常优秀 TTS 系统 MCD < 8 dB

采样率自动参数：

| 采样率 | mcep_dim | mcep_alpha |
|--------|----------|------------|
| 16000 Hz | 23 | 0.42 |
| 22050 Hz | 34 | 0.45 |
| 24000 Hz | 34 | 0.46 |
| 44100 Hz | 39 | 0.53 |
| 48000 Hz | 39 | 0.55 |

### 使用

```python
from tts import compute_mcd_pair, MCD

# 快捷函数
mcd = compute_mcd_pair("generated.wav", "ground_truth.wav")
print(f"MCD: {mcd:.4f} dB")

# 类接口
metric = MCD(n_fft=1024, n_shift=256)
result = metric.compute("gen.wav", "gt.wav")
print(f"MCD: {result['mcd']} dB")

# 批量
result = metric.compute_batch(gen_paths, gt_paths)
print(f"Mean MCD: {result['mean_mcd']} ± {result['std_mcd']} dB")

# 目录评测
result = metric.compute_from_dir("generated/", "ground_truth/", output_dir="results/")
# 生成 results/utt2mcd 和 results/mcd_avg_result.txt
```

## Log-F0 RMSE

### 原理

$$\text{Log-F0 RMSE} = \sqrt{\frac{1}{N}\sum_{n=1}^{N}(\log F0^{gen}_n - \log F0^{gt}_n)^2}$$

- 使用 World vocoder (HARVEST) 提取 F0
- DTW 对齐梅尔倒谱帧序列
- 仅在双方都有声（F0 > 0）的帧上计算
- 取对数消除绝对频率的尺度影响

### 使用

```python
from tts import compute_f0_rmse_pair, F0RMSE

# 快捷函数
rmse = compute_f0_rmse_pair("generated.wav", "ground_truth.wav")
print(f"Log-F0 RMSE: {rmse:.4f}")

# 类接口
metric = F0RMSE(f0min=40, f0max=800)
result = metric.compute("gen.wav", "gt.wav")
print(f"Log-F0 RMSE: {result['log_f0_rmse']}")

# 批量
result = metric.compute_batch(gen_paths, gt_paths)
print(f"Mean: {result['mean_log_f0_rmse']} ± {result['std_log_f0_rmse']}")

# 目录评测
result = metric.compute_from_dir("generated/", "ground_truth/", output_dir="results/")
# 生成 results/utt2log_f0_rmse 和 results/log_f0_rmse_avg_result.txt
```

## SpeechBERTScore

### 原理

- 基于预训练语音模型 WavLM-Large 第 14 层提取语音表征
- 参照 BERTScore 方法计算余弦相似度
- 返回 score（F1）、precision、recall 三个分量
- 论文：[arXiv:2401.16812](https://arxiv.org/abs/2401.16812)
- 适用于 TTS、语音转换、语音增强等任务

### 使用

```python
from tts import compute_speech_bert_score, SpeechBERTScoreMetric

# 快捷函数
score = compute_speech_bert_score("generated.wav", "ground_truth.wav")
print(f"SpeechBERTScore: {score}")

# 类接口（首次运行下载 WavLM-Large ~1.2GB）
metric = SpeechBERTScoreMetric(model_type="wavlm-large", layer=14, use_gpu=True)

# 单对 - 返回 score/precision/recall
result = metric.compute("gen.wav", "gt.wav")
print(f"Score: {result['speech_bert_score']}")
print(f"Precision: {result['precision']}")
print(f"Recall: {result['recall']}")

# 批量
result = metric.compute_batch(gen_paths, gt_paths)
print(f"Mean: {result['mean_score']} ± {result['std_score']}")

# 目录评测
result = metric.compute_from_dir("generated/", "ground_truth/", output_dir="results/")
# 生成 results/utt2spbs 和 results/spbs_avg_result.txt
```

## SpeechBLEU

### 原理

- 基于 HuBERT-Base 第 11 层将语音离散化为 token 序列（vocab=200）
- 去除连续重复 token 后，计算 BLEU 风格的 n-gram 匹配分数
- 论文：[arXiv:2401.16812](https://arxiv.org/abs/2401.16812)
- 衡量生成语音与真实语音在离散语义单元上的结构相似性

### 使用

```python
from tts import compute_speech_bleu, SpeechBLEUMetric

# 快捷函数
score = compute_speech_bleu("generated.wav", "ground_truth.wav")
print(f"SpeechBLEU: {score}")

# 类接口（首次运行下载 HuBERT 模型）
metric = SpeechBLEUMetric(
    model_type="hubert-base",
    vocab=200,
    layer=11,
    n_ngram=2,
    remove_repetition=True,
    use_gpu=True,
)

# 单对
result = metric.compute("gen.wav", "gt.wav")
print(f"SpeechBLEU: {result['speech_bleu']}")

# 批量
result = metric.compute_batch(gen_paths, gt_paths)
print(f"Mean: {result['mean_score']} ± {result['std_score']}")

# 目录评测
result = metric.compute_from_dir("generated/", "ground_truth/", output_dir="results/")
# 生成 results/utt2spbleu 和 results/spbleu_avg_result.txt
```

## VUV Error (清浊音错误率)

### 原理

$$\text{VUV Error} = \frac{\text{清浊音不匹配帧数}}{\text{总对齐帧数}} \times 100\%$$

- 使用 World vocoder (HARVEST) 提取 F0
- 基于梅尔倒谱通过 DTW 对齐帧序列
- F0 > 0 为有声帧（Voiced），F0 == 0 为无声帧（Unvoiced）
- 逐帧比较清浊音决策是否一致
- TTS 和 SVS 通用指标，代码在 `common/vuv.py`

### 使用

```python
from tts import VUVError, compute_vuv_error_pair

# 快捷函数
result = compute_vuv_error_pair("generated.wav", "ground_truth.wav")
print(f"VUV Error: {result['vuv_error_pct']}%")

# 类接口
metric = VUVError(f0min=40, f0max=800)
result = metric.compute("gen.wav", "gt.wav")
print(f"VUV Error: {result['vuv_error_pct']}%")
print(f"Error frames: {result['error_frames']} / {result['total_frames']}")

# 批量
result = metric.compute_batch(gen_paths, gt_paths)
print(f"Mean VUV Error: {result['mean_vuv_error_pct']}%")

# 目录评测
result = metric.compute_from_dir("generated/", "ground_truth/", output_dir="results/")
# 生成 results/utt2vuv_error 和 results/vuv_error_avg_result.txt
```

## API Reference

### MCD

| 接口 | 返回 | 说明 |
|------|------|------|
| `compute_mcd_pair(gen, gt, ...)` | float | 单对 MCD (dB) |
| `MCD.compute(gen, gt)` | Dict | `{"mcd": float}` |
| `MCD.compute_batch(gens, gts)` | Dict | mean_mcd, std_mcd, details |
| `MCD.compute_from_dir(gen_dir, gt_dir, output_dir)` | Dict | 目录批量 + 结果文件 |

### F0RMSE

| 接口 | 返回 | 说明 |
|------|------|------|
| `compute_f0_rmse_pair(gen, gt, ...)` | float | 单对 log-F0 RMSE |
| `F0RMSE.compute(gen, gt)` | Dict | `{"log_f0_rmse": float}` |
| `F0RMSE.compute_batch(gens, gts)` | Dict | mean/std + details |
| `F0RMSE.compute_from_dir(gen_dir, gt_dir, output_dir)` | Dict | 目录批量 + 结果文件 |

### SpeechBERTScoreMetric

| 接口 | 返回 | 说明 |
|------|------|------|
| `compute_speech_bert_score(gen, gt, ...)` | float | 单对 score |
| `SpeechBERTScoreMetric.compute(gen, gt)` | Dict | score, precision, recall |
| `SpeechBERTScoreMetric.compute_batch(gens, gts)` | Dict | mean/std + details |
| `SpeechBERTScoreMetric.compute_from_dir(...)` | Dict | 目录批量 + 结果文件 |

### SpeechBLEU

| 接口 | 返回 | 说明 |
|------|------|------|
| `compute_speech_bleu(gen, gt, ...)` | float | 单对 SpeechBLEU 分数 |
| `SpeechBLEUMetric.compute(gen, gt)` | Dict | `{"speech_bleu": float}` |
| `SpeechBLEUMetric.compute_batch(gens, gts)` | Dict | mean/std + details |
| `SpeechBLEUMetric.compute_from_dir(...)` | Dict | 目录批量 + 结果文件 |

### VUVError

| 接口 | 返回 | 说明 |
|------|------|------|
| `compute_vuv_error_pair(gen, gt, ...)` | Dict | vuv_error, vuv_error_pct, total_frames, error_frames |
| `VUVError.compute(gen, gt)` | Dict | 同上 |
| `VUVError.compute_batch(gens, gts)` | Dict | mean/std + details |
| `VUVError.compute_from_dir(gen_dir, gt_dir, output_dir)` | Dict | 目录批量 + 结果文件 |

## 参考来源

- MCD：[ESPnet - evaluate_mcd.py](https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/asr1/pyscripts/utils/evaluate_mcd.py)
- F0 RMSE：[ESPnet - evaluate_f0.py](https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/asr1/pyscripts/utils/evaluate_f0.py)
- SpeechBERTScore：[ESPnet - evaluate_speechbertscore.py](https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/asr1/pyscripts/utils/evaluate_speechbertscore.py)
- SpeechBLEU：[ESPnet - evaluate_speechbleu.py](https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/asr1/pyscripts/utils/evaluate_speechbleu.py)
- 底层库：[pysptk](https://github.com/r9y9/pysptk)、[pyworld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder)、[fastdtw](https://github.com/slaypni/fastdtw)、[discrete-speech-metrics](https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics)
