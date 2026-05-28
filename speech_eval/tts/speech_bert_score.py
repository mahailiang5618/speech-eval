"""SpeechBERTScore metric for TTS evaluation.

Inspired by ESPnet's evaluate_speechbertscore.py, measures speech quality
using discrete speech representations from WavLM-Large model.
Reference: https://arxiv.org/abs/2401.16812

Uses the discrete-speech-metrics package:
https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics
"""

from __future__ import annotations

import fnmatch
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

try:
    import librosa
except ImportError:
    librosa = None

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from discrete_speech_metrics import SpeechBERTScore as _SpeechBERTScore
except ImportError:
    _SpeechBERTScore = None


def _check_dependencies():
    missing = []
    if librosa is None:
        missing.append("librosa")
    if sf is None:
        missing.append("soundfile")
    if _SpeechBERTScore is None:
        missing.append("discrete-speech-metrics")
    if missing:
        raise RuntimeError(
            f"Missing dependencies for SpeechBERTScore: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
        )


TARGET_SR = 16000


def _load_and_normalize(path: Union[str, Path], target_sr: int = TARGET_SR) -> np.ndarray:
    """Load audio, resample to target_sr, and normalize amplitude to [-1, 1]."""
    x, fs = sf.read(str(path), dtype="int16")
    if fs != target_sr:
        x = librosa.resample(x.astype(np.float64), orig_sr=fs, target_sr=target_sr)
    x = x.astype(np.float32)
    amax = np.amax(np.absolute(x))
    if amax > 0:
        x = x / amax
    return x


def find_audio_files(root_dir: str, extensions: List[str] = None) -> List[str]:
    """Recursively find audio files in a directory."""
    if extensions is None:
        extensions = ["*.wav", "*.flac"]
    files = []
    for root, _, filenames in os.walk(root_dir, followlinks=True):
        for ext in extensions:
            for filename in fnmatch.filter(filenames, ext):
                files.append(os.path.join(root, filename))
    return sorted(files)


class SpeechBERTScoreMetric:
    """Compute SpeechBERTScore for TTS evaluation.

    SpeechBERTScore uses WavLM-Large to extract speech representations and
    computes a BERTScore-style metric between generated and reference audio.
    Higher scores indicate better speech quality/similarity.

    Args:
        model_type: Model to use for feature extraction (default: "wavlm-large").
        layer: Which layer's representations to use (default: 14, best per paper).
        use_gpu: Whether to use GPU for computation.
        sr: Target sampling rate (default: 16000).
    """

    def __init__(
        self,
        model_type: str = "wavlm-large",
        layer: int = 14,
        use_gpu: bool = True,
        sr: int = TARGET_SR,
    ) -> None:
        _check_dependencies()
        self.sr = sr
        self.model_type = model_type
        self.layer = layer
        self.use_gpu = use_gpu
        self._metrics = _SpeechBERTScore(
            sr=sr, model_type=model_type, layer=layer, use_gpu=use_gpu
        )

    def compute(
        self,
        gen_path: Union[str, Path],
        gt_path: Union[str, Path],
    ) -> Dict[str, float]:
        """Compute SpeechBERTScore for a single audio pair.

        Args:
            gen_path: Path to generated audio file.
            gt_path: Path to ground truth audio file.

        Returns:
            Dict with:
                - "speech_bert_score": the overall score (higher is better)
                - "precision": precision component
                - "recall": recall component
        """
        gen_x = _load_and_normalize(gen_path, self.sr)
        gt_x = _load_and_normalize(gt_path, self.sr)

        score, precision, recall = self._metrics.score(gt_x, gen_x)

        return {
            "speech_bert_score": round(float(score), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
        }

    def compute_batch(
        self,
        gen_paths: List[Union[str, Path]],
        gt_paths: List[Union[str, Path]],
    ) -> Dict[str, object]:
        """Compute SpeechBERTScore for multiple audio pairs.

        Args:
            gen_paths: List of generated audio file paths.
            gt_paths: List of ground truth audio file paths.

        Returns:
            Dict with "mean_score", "std_score", "num_utterances", and "details".
        """
        assert len(gen_paths) == len(gt_paths), (
            f"Mismatch: {len(gen_paths)} generated vs {len(gt_paths)} ground truth files"
        )

        results = {}
        for gen_path, gt_path in zip(gen_paths, gt_paths):
            basename = Path(gen_path).stem
            gen_x = _load_and_normalize(gen_path, self.sr)
            gt_x = _load_and_normalize(gt_path, self.sr)
            score, _, _ = self._metrics.score(gt_x, gen_x)
            results[basename] = round(float(score), 4)
            logging.info(f"{basename} {score:.4f}")

        values = list(results.values())
        return {
            "mean_score": round(float(np.mean(values)), 4),
            "std_score": round(float(np.std(values)), 4),
            "num_utterances": len(values),
            "details": results,
        }

    def compute_from_dir(
        self,
        gen_dir: Union[str, Path],
        gt_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, object]:
        """Compute SpeechBERTScore for all matching audio files between two directories.

        Files are matched by basename (filename without extension).

        Args:
            gen_dir: Directory of generated audio files.
            gt_dir: Directory of ground truth audio files.
            output_dir: If provided, write results to this directory.

        Returns:
            Dict with mean/std SpeechBERTScore and per-utterance details.
        """
        gen_files = find_audio_files(str(gen_dir))
        gt_files = find_audio_files(str(gt_dir))

        if not gen_files:
            raise FileNotFoundError(f"No audio files found in {gen_dir}")
        if not gt_files:
            raise FileNotFoundError(f"No audio files found in {gt_dir}")

        gt_map = {Path(f).stem: f for f in gt_files}

        matched_gen = []
        matched_gt = []
        for gen_path in gen_files:
            basename = Path(gen_path).stem
            if basename in gt_map:
                matched_gen.append(gen_path)
                matched_gt.append(gt_map[basename])

        if not matched_gen:
            raise ValueError(
                f"No matching files between gen_dir and gt_dir. "
                f"gen has {len(gen_files)} files, gt has {len(gt_files)} files."
            )

        logging.info(f"Found {len(matched_gen)} matched utterances")

        result = self.compute_batch(matched_gen, matched_gt)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_dir / "utt2spbs", "w") as f:
                for utt_id in sorted(result["details"].keys()):
                    f.write(f"{utt_id} {result['details'][utt_id]:.4f}\n")

            with open(output_dir / "spbs_avg_result.txt", "w") as f:
                f.write(f"#utterances: {result['num_utterances']}\n")
                f.write(f"Average: {result['mean_score']:.4f} +/- {result['std_score']:.4f}\n")

            result["output_dir"] = str(output_dir)

        return result


def compute_speech_bert_score(
    gen_path: Union[str, Path],
    gt_path: Union[str, Path],
    model_type: str = "wavlm-large",
    layer: int = 14,
    use_gpu: bool = True,
) -> float:
    """Quick function to compute SpeechBERTScore for a single pair.

    Args:
        gen_path: Path to generated audio.
        gt_path: Path to ground truth audio.
        model_type: WavLM model type.
        layer: Layer to extract features from.
        use_gpu: Whether to use GPU.

    Returns:
        SpeechBERTScore value (higher is better, range roughly 0~1).
    """
    _check_dependencies()

    gen_x = _load_and_normalize(gen_path)
    gt_x = _load_and_normalize(gt_path)

    metrics = _SpeechBERTScore(sr=TARGET_SR, model_type=model_type, layer=layer, use_gpu=use_gpu)
    score, _, _ = metrics.score(gt_x, gen_x)
    return round(float(score), 4)
