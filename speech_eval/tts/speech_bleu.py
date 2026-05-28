"""SpeechBLEU metric for TTS evaluation.

Inspired by ESPnet's evaluate_speechbleu.py, measures speech quality using
discrete speech tokens from HuBERT and computing BLEU-style n-gram overlap.
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
    from discrete_speech_metrics import SpeechBLEU as _SpeechBLEU
except ImportError:
    _SpeechBLEU = None


def _check_dependencies():
    missing = []
    if librosa is None:
        missing.append("librosa")
    if sf is None:
        missing.append("soundfile")
    if _SpeechBLEU is None:
        missing.append("discrete-speech-metrics")
    if missing:
        raise RuntimeError(
            f"Missing dependencies for SpeechBLEU: {', '.join(missing)}. "
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


class SpeechBLEUMetric:
    """Compute SpeechBLEU for TTS evaluation.

    SpeechBLEU uses HuBERT to discretize speech into tokens and computes
    BLEU-style n-gram overlap between generated and reference audio.
    Higher scores indicate better speech quality/similarity.

    Args:
        model_type: Model for token extraction (default: "hubert-base").
        vocab: Vocabulary size for discretization (default: 200).
        layer: Which layer's representations to use (default: 11).
        n_ngram: N-gram order for BLEU computation (default: 2).
        remove_repetition: Whether to remove consecutive repeated tokens (default: True).
        use_gpu: Whether to use GPU for computation.
        sr: Target sampling rate (default: 16000).
    """

    def __init__(
        self,
        model_type: str = "hubert-base",
        vocab: int = 200,
        layer: int = 11,
        n_ngram: int = 2,
        remove_repetition: bool = True,
        use_gpu: bool = True,
        sr: int = TARGET_SR,
    ) -> None:
        _check_dependencies()
        self.sr = sr
        self._metrics = _SpeechBLEU(
            sr=sr,
            model_type=model_type,
            vocab=vocab,
            layer=layer,
            n_ngram=n_ngram,
            remove_repetition=remove_repetition,
            use_gpu=use_gpu,
        )

    def compute(
        self,
        gen_path: Union[str, Path],
        gt_path: Union[str, Path],
    ) -> Dict[str, float]:
        """Compute SpeechBLEU for a single audio pair.

        Args:
            gen_path: Path to generated audio file.
            gt_path: Path to ground truth audio file.

        Returns:
            Dict with "speech_bleu" score (higher is better).
        """
        gen_x = _load_and_normalize(gen_path, self.sr)
        gt_x = _load_and_normalize(gt_path, self.sr)

        score = self._metrics.score(gt_x, gen_x)

        return {"speech_bleu": round(float(score), 4)}

    def compute_batch(
        self,
        gen_paths: List[Union[str, Path]],
        gt_paths: List[Union[str, Path]],
    ) -> Dict[str, object]:
        """Compute SpeechBLEU for multiple audio pairs.

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
            score = self._metrics.score(gt_x, gen_x)
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
        """Compute SpeechBLEU for all matching audio files between two directories.

        Files are matched by basename (filename without extension).

        Args:
            gen_dir: Directory of generated audio files.
            gt_dir: Directory of ground truth audio files.
            output_dir: If provided, write results to this directory.

        Returns:
            Dict with mean/std SpeechBLEU and per-utterance details.
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

            with open(output_dir / "utt2spbleu", "w") as f:
                for utt_id in sorted(result["details"].keys()):
                    f.write(f"{utt_id} {result['details'][utt_id]:.4f}\n")

            with open(output_dir / "spbleu_avg_result.txt", "w") as f:
                f.write(f"#utterances: {result['num_utterances']}\n")
                f.write(f"Average: {result['mean_score']:.4f} +/- {result['std_score']:.4f}\n")

            result["output_dir"] = str(output_dir)

        return result


def compute_speech_bleu(
    gen_path: Union[str, Path],
    gt_path: Union[str, Path],
    model_type: str = "hubert-base",
    vocab: int = 200,
    layer: int = 11,
    n_ngram: int = 2,
    remove_repetition: bool = True,
    use_gpu: bool = True,
) -> float:
    """Quick function to compute SpeechBLEU for a single pair.

    Args:
        gen_path: Path to generated audio.
        gt_path: Path to ground truth audio.
        model_type: HuBERT model type.
        vocab: Vocabulary size for discretization.
        layer: Layer to extract features from.
        n_ngram: N-gram order.
        remove_repetition: Remove consecutive repeated tokens.
        use_gpu: Whether to use GPU.

    Returns:
        SpeechBLEU score (higher is better).
    """
    _check_dependencies()

    gen_x = _load_and_normalize(gen_path)
    gt_x = _load_and_normalize(gt_path)

    metrics = _SpeechBLEU(
        sr=TARGET_SR,
        model_type=model_type,
        vocab=vocab,
        layer=layer,
        n_ngram=n_ngram,
        remove_repetition=remove_repetition,
        use_gpu=use_gpu,
    )
    score = metrics.score(gt_x, gen_x)
    return round(float(score), 4)
