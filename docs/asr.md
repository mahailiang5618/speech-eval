# ASR 评测指标

自动语音识别（ASR）评测指标实现，包含 WER 和 CER。

## 指标概览

| 指标 | 说明 | 适用场景 |
|------|------|----------|
| WER (Word Error Rate) | 词错误率 | 英文 ASR |
| CER (Character Error Rate) | 字错误率 | 中文 ASR |

## 安装依赖

```bash
pip install jiwer>=3.0.0
```

## 快速开始

### WER

```python
from asr import compute_wer

ref = "the cat sat on the mat"
hyp = "the cat sit on a mat"
wer = compute_wer(ref, hyp)
print(f"WER: {wer}%")  # WER: 33.33%
```

### CER

```python
from asr import compute_cer

ref = "今天天气很好"
hyp = "今天气很好"
cer = compute_cer(ref, hyp)
print(f"CER: {cer}%")  # CER: 16.67%
```

## 使用方式

### 方式一：快捷函数

```python
from asr import compute_wer, compute_cer

# 单条
wer = compute_wer(ref, hyp)
cer = compute_cer(ref, hyp)

# 批量
wer = compute_wer(ref_list, hyp_list)
```

### 方式二：类接口 - 基础计算

```python
from asr import WER

metric = WER(lowercase=True, remove_punctuation=True)
result = metric.compute(
    reference="The quick brown fox jumps over the lazy dog",
    hypothesis="the quick brown fox jump over a lazy dog",
)
print(f"WER: {result['wer']}%")              # 22.22%
print(f"Substitutions: {result['substitutions']}")  # 2
print(f"Deletions: {result['deletions']}")          # 0
print(f"Insertions: {result['insertions']}")        # 0
```

### 方式三：详细结果 + 对齐可视化

```python
from asr import WER

metric = WER()
result = metric.compute_detail(
    reference=["i have a dream", "the weather is nice today"],
    hypothesis=["i have dream", "the weather is nice today"],
)
print(f"Overall WER: {result['wer']}%")
print(f"Per-sentence WER: {result['sentence_wers']}")
print(result['alignment'])
```

输出：
```
=== SENTENCE 1 ===
REF: i have a dream
HYP: i have * dream
            D
```

### 方式四：基于文件评测

```python
from asr import WER

metric = WER()
result = metric.compute_from_file(
    ref_path="data/ref.txt",    # 每行一条 reference
    hyp_path="data/hyp.txt",    # 每行一条 hypothesis（与 ref 逐行对齐）
    output_dir="results/",      # 可选，保存对齐可视化文件
)
print(f"WER: {result['wer']}%")
```

### CER 类使用（接口同 WER）

```python
from asr import CER

metric = CER()
result = metric.compute("北京市海淀区中关村大街", "北京市海淀中关村大街")
print(f"CER: {result['cer']}%")          # 9.09%
print(f"Deletions: {result['deletions']}")  # 1

# 批量 + 对齐可视化
result = metric.compute_detail(
    reference=["人工智能改变世界", "机器学习算法优化"],
    hypothesis=["人工智能改变时界", "机器学习算法优化"],
)
print(f"Overall CER: {result['cer']}%")     # 6.25%
print(result['alignment'])
```

## 评估指标详解

### 基础错误统计

| 统计量 | 含义 | 说明 |
|--------|------|------|
| Substitutions (S) | 替换数 | reference 中的词/字被识别成了另一个词/字 |
| Deletions (D) | 删除数 | reference 中有的词/字在 hypothesis 中缺失 |
| Insertions (I) | 插入数 | hypothesis 中多出了 reference 没有的词/字 |
| Hits (H) | 命中数 | 正确识别的词/字数 |

### WER

$$WER = \frac{S + D + I}{N} \times 100\%$$

- N = 参考文本总词数（N = S + D + H）
- 最常用的 ASR 评测主指标，越低越好

### CER

$$CER = \frac{S + D + I}{N} \times 100\%$$

- 计算方式与 WER 相同，但以字符为单位
- 适用于中文、日文等无天然词边界的语言

### MER (Match Error Rate)

$$MER = \frac{S + D + I}{S + D + I + H} \times 100\%$$

- 分母为对齐后的总匹配数（含错误 + 命中）
- 与 WER 类似，当无插入错误时两者相等

### WIL (Word Information Lost)

$$WIL = 1 - \frac{H^2}{N \times P} \times 100\%$$

- N = reference 词数，P = hypothesis 词数
- 衡量 reference 与 hypothesis 之间的信息偏差程度，越低越好

### WIP (Word Information Preserved)

$$WIP = 1 - WIL$$

- WIL 的互补指标，表示信息保留率，越高越好

### 指标选择建议

| 场景 | 推荐指标 | 原因 |
|------|----------|------|
| 英文 ASR 评测 | WER | 业界标准，论文可比 |
| 中文 ASR 评测 | CER | 中文无天然词边界，字级别更准确 |
| 关注信息完整性 | WIL/WIP | 同时考虑 ref 和 hyp 长度的影响 |
| 严格匹配评估 | MER | 对插入错误更敏感 |

## API Reference

### `compute_wer(reference, hypothesis, lowercase=True, remove_punctuation=True) -> float`

快捷函数，直接返回 WER 百分比值。

| 参数 | 类型 | 说明 |
|------|------|------|
| reference | str 或 List[str] | 参考文本 |
| hypothesis | str 或 List[str] | 识别结果文本 |
| lowercase | bool | 是否转小写（默认 True） |
| remove_punctuation | bool | 是否去标点（默认 True） |

### `WER` 类

| 方法 | 返回值 | 说明 |
|------|--------|------|
| `compute(ref, hyp)` | Dict | 返回 wer、substitutions、deletions、insertions、ref_word_count |
| `compute_detail(ref, hyp)` | Dict | 额外返回 sentence_wers 和 alignment 可视化字符串 |
| `compute_from_file(ref_path, hyp_path, output_dir)` | Dict | 从文件读取并计算，可选导出对齐结果 |

### `compute_cer(reference, hypothesis, lowercase=True, remove_punctuation=True) -> float`

快捷函数，直接返回 CER 百分比值。参数同 `compute_wer`。

### `CER` 类

| 方法 | 返回值 | 说明 |
|------|--------|------|
| `compute(ref, hyp)` | Dict | 返回 cer、substitutions、deletions、insertions、ref_char_count |
| `compute_detail(ref, hyp)` | Dict | 额外返回 sentence_cers 和 alignment 可视化字符串 |
| `compute_from_file(ref_path, hyp_path, output_dir)` | Dict | 从文件读取并计算，可选导出对齐结果 |

## 参考来源

- WER：[ESPnet - espnet3/systems/asr/metrics/wer.py](https://github.com/espnet/espnet/blob/master/espnet3/systems/asr/metrics/wer.py)
- CER：[ESPnet - espnet3/systems/asr/metrics/cer.py](https://github.com/espnet/espnet/blob/master/espnet3/systems/asr/metrics/cer.py)
- 底层库：[jiwer](https://github.com/jitsi/jiwer)
