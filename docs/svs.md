# SVS 评测指标

歌声合成（Singing Voice Synthesis）评测指标实现，包含 Semitone ACC。

## 指标概览

| 指标 | 说明 | 方向 |
|------|------|------|
| Semitone ACC | 半音准确率，音符级别的音高准确度 | 越高越好 |
| VUV Error | 清浊音帧级分类错误率 | 越低越好 |

## 安装依赖

```bash
pip install librosa pysptk pyworld soundfile fastdtw scipy
```

## Semitone ACC (半音准确率)

### 原理

$$\text{Semitone ACC} = \frac{\text{音符匹配帧数}}{\text{总帧数}} \times 100\%$$

- 使用 World vocoder 提取 F0，通过 DTW 对齐帧序列
- 将每帧 F0 (Hz) 转换为音乐半音音符（如 C4、A#3、Sil）
- 逐帧比较合成与真实音频的音符是否一致
- 100% 表示每一帧音符都完全正确

音符转换公式：$h = \text{round}(12 \cdot \log_2(f / C_0))$，其中 $C_0 = 440 \times 2^{-4.75}$ Hz

### 工具函数：Hz → 半音音符

```python
from svs import hz_to_semitone

print(hz_to_semitone(440.0))   # A_4
print(hz_to_semitone(261.63))  # C_4
print(hz_to_semitone(0))       # Sil (静音)
```

### 快捷函数

```python
from svs import compute_semitone_acc_pair

result = compute_semitone_acc_pair("generated_singing.wav", "ground_truth.wav")
print(f"Semitone ACC: {result['semitone_acc_pct']}%")
print(f"Total frames: {result['total_frames']}")
```

### SemitoneACC 类

```python
from svs import SemitoneACC

metric = SemitoneACC(f0min=40, f0max=800)

# 单对
result = metric.compute("gen.wav", "gt.wav")
print(f"Semitone ACC: {result['semitone_acc_pct']}%")

# 批量
result = metric.compute_batch(
    gen_paths=["gen_001.wav", "gen_002.wav"],
    gt_paths=["gt_001.wav", "gt_002.wav"],
)
print(f"Mean Semitone ACC: {result['mean_semitone_acc_pct']}%")

# 目录评测
result = metric.compute_from_dir(
    gen_dir="output/generated_singing/",
    gt_dir="data/ground_truth_singing/",
    output_dir="results/",
)
print(f"Mean Semitone ACC: {result['mean_semitone_acc_pct']}%")
# 生成 results/utt2semitone_acc 和 results/semitone_acc_avg_result.txt
```

## VUV Error (清浊音错误率)

TTS 和 SVS 通用指标，代码位于 `common/vuv.py`，通过 svs 模块可直接使用。

### 原理

$$\text{VUV Error} = \frac{\text{清浊音不匹配帧数}}{\text{总对齐帧数}} \times 100\%$$

- F0 > 0 为有声帧（Voiced），F0 == 0 为无声帧（Unvoiced）
- 基于梅尔倒谱 DTW 对齐后，逐帧比较清浊音决策

### 使用

```python
from svs import VUVError, compute_vuv_error_pair

# 快捷函数
result = compute_vuv_error_pair("generated_singing.wav", "ground_truth.wav")
print(f"VUV Error: {result['vuv_error_pct']}%")

# 类接口
metric = VUVError(f0min=40, f0max=800)
result = metric.compute("gen.wav", "gt.wav")
print(f"VUV Error: {result['vuv_error_pct']}%")

# 目录评测
result = metric.compute_from_dir("generated/", "ground_truth/", output_dir="results/")
# 生成 results/utt2vuv_error 和 results/vuv_error_avg_result.txt
```

## API Reference

| 接口 | 返回 | 说明 |
|------|------|------|
| `hz_to_semitone(freq)` | str | Hz 转半音音符名（如 "A_4"、"Sil"） |
| `compute_semitone_acc_pair(gen, gt, ...)` | Dict | semitone_acc, semitone_acc_pct, total_frames |
| `SemitoneACC.compute(gen, gt)` | Dict | 同上 |
| `SemitoneACC.compute_batch(gens, gts)` | Dict | mean/std + details |
| `SemitoneACC.compute_from_dir(gen_dir, gt_dir, output_dir)` | Dict | 目录批量 + 结果文件 |
| `compute_vuv_error_pair(gen, gt, ...)` | Dict | vuv_error, vuv_error_pct, total_frames, error_frames |
| `VUVError.compute(gen, gt)` | Dict | 同上 |
| `VUVError.compute_batch(gens, gts)` | Dict | mean/std + details |
| `VUVError.compute_from_dir(gen_dir, gt_dir, output_dir)` | Dict | 目录批量 + 结果文件 |

## 参考来源

- Semitone ACC：[ESPnet - evaluate_semitone.py](https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/asr1/pyscripts/utils/evaluate_semitone.py)
- VUV Error：[ESPnet - evaluate_vuv.py](https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/asr1/pyscripts/utils/evaluate_vuv.py)
- 底层库：[pysptk](https://github.com/r9y9/pysptk)、[pyworld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder)、[fastdtw](https://github.com/slaypni/fastdtw)
