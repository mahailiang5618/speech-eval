"""Voiced/Unvoiced (VUV) Error metric for TTS and SVS evaluation.

Inspired by ESPnet's evaluate_vuv.py, measures the frame-level voiced/unvoiced
classification accuracy between generated and ground truth audio.
Applicable to both TTS and SVS tasks.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import librosa
except ImportError:
    librosa = None

try:
    import pysptk
except ImportError:
    pysptk = None

try:
    import pyworld as pw
except ImportError:
    pw = None

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from fastdtw import fastdtw
except ImportError:
    fastdtw = None

from scipy import spatial


def _check_dependencies():
    missing = []
    if librosa is None:
        missing.append("librosa")
    if pysptk is None:
        missing.append("pysptk")
    if pw is None:
        missing.append("pyworld")
    if sf is None:
        missing.append("soundfile")
    if fastdtw is None:
        missing.append("fastdtw")
    if missing:
        raise RuntimeError(
            f"Missing dependencies for VUV Error: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
        )


def _get_best_mcep_params(fs: int) -> Tuple[int, float]:
    params = {
        16000: (23, 0.42),
        22050: (34, 0.45),
        24000: (34, 0.46),
        44100: (39, 0.53),
        48000: (39, 0.55),
    }
    if fs not in params:
        raise ValueError(
            f"No preset mcep params for fs={fs}. Supported: {list(params.keys())}."
        )
    return params[fs]


def world_extract(
    x: np.ndarray,
    fs: int,
    f0min: int = 40,
    f0max: int = 800,
    n_fft: int = 1024,
    n_shift: int = 256,
    mcep_dim: Optional[int] = None,
    mcep_alpha: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract World-based mel-cepstrum and F0."""
    x = x.astype(np.float64)
    f0, time_axis = pw.harvest(
        x, fs,
        f0_floor=f0min,
        f0_ceil=f0max,
        frame_period=n_shift / fs * 1000,
    )
    sp = pw.cheaptrick(x, f0, time_axis, fs, fft_size=n_fft)

    if mcep_dim is None or mcep_alpha is None:
        mcep_dim, mcep_alpha = _get_best_mcep_params(fs)
    mcep = pysptk.sp2mc(sp, mcep_dim, mcep_alpha)

    return mcep, f0


def compute_vuv_error_pair(
    gen_path: Union[str, Path],
    gt_path: Union[str, Path],
    f0min: int = 40,
    f0max: int = 800,
    n_fft: int = 1024,
    n_shift: int = 256,
    mcep_dim: Optional[int] = None,
    mcep_alpha: Optional[float] = None,
) -> Dict[str, float]:
    """Compute VUV Error between a single pair of audio files.

    VUV Error measures the percentage of frames where the voiced/unvoiced
    decision differs between generated and ground truth audio.

    Args:
        gen_path: Path to generated audio.
        gt_path: Path to ground truth audio.
        f0min: Minimum F0 in Hz.
        f0max: Maximum F0 in Hz.
        n_fft: FFT length.
        n_shift: Shift length.
        mcep_dim: Mel-cepstrum dimension (auto if None).
        mcep_alpha: All-pass coefficient (auto if None).

    Returns:
        Dict with:
            - "vuv_error": error rate as ratio (0~1)
            - "vuv_error_pct": error rate as percentage
            - "total_frames": total aligned frames
            - "error_frames": number of mismatched frames
    """
    _check_dependencies()

    gen_x, gen_fs = sf.read(str(gen_path), dtype="int16")
    gt_x, gt_fs = sf.read(str(gt_path), dtype="int16")

    fs = gen_fs
    if gen_fs != gt_fs:
        gt_x = librosa.resample(
            gt_x.astype(np.float64), orig_sr=gt_fs, target_sr=gen_fs
        ).astype(np.int16)

    gen_mcep, gen_f0 = world_extract(
        x=gen_x, fs=fs, f0min=f0min, f0max=f0max,
        n_fft=n_fft, n_shift=n_shift,
        mcep_dim=mcep_dim, mcep_alpha=mcep_alpha,
    )
    gt_mcep, gt_f0 = world_extract(
        x=gt_x, fs=fs, f0min=f0min, f0max=f0max,
        n_fft=n_fft, n_shift=n_shift,
        mcep_dim=mcep_dim, mcep_alpha=mcep_alpha,
    )

    # DTW alignment
    _, path = fastdtw(gen_mcep, gt_mcep, dist=spatial.distance.euclidean)
    twf = np.array(path).T
    gen_f0_dtw = gen_f0[twf[0]]
    gt_f0_dtw = gt_f0[twf[1]]

    # VUV classification: voiced (F0 > 0) vs unvoiced (F0 == 0)
    vuv_gen = gen_f0_dtw != 0
    vuv_gt = gt_f0_dtw != 0

    error_frames = int((vuv_gen != vuv_gt).sum())
    total_frames = len(vuv_gt)
    vuv_error = error_frames / total_frames

    return {
        "vuv_error": round(vuv_error, 4),
        "vuv_error_pct": round(vuv_error * 100, 2),
        "total_frames": total_frames,
        "error_frames": error_frames,
    }


def find_audio_files(root_dir: str, extensions: List[str] = None) -> List[str]:
    if extensions is None:
        extensions = ["*.wav", "*.flac"]
    files = []
    for root, _, filenames in os.walk(root_dir, followlinks=True):
        for ext in extensions:
            for filename in fnmatch.filter(filenames, ext):
                files.append(os.path.join(root, filename))
    return sorted(files)


class VUVError:
    """Compute Voiced/Unvoiced Error for TTS and SVS evaluation.

    Measures the frame-level accuracy of voiced/unvoiced classification.
    A frame is "voiced" if F0 > 0, "unvoiced" if F0 == 0.

    Args:
        f0min: Minimum F0 value in Hz.
        f0max: Maximum F0 value in Hz.
        n_fft: FFT length in points.
        n_shift: Shift length in points.
        mcep_dim: Mel-cepstrum dimension (auto if None).
        mcep_alpha: All-pass filter coefficient (auto if None).
    """

    def __init__(
        self,
        f0min: int = 40,
        f0max: int = 800,
        n_fft: int = 1024,
        n_shift: int = 256,
        mcep_dim: Optional[int] = None,
        mcep_alpha: Optional[float] = None,
    ) -> None:
        _check_dependencies()
        self.f0min = f0min
        self.f0max = f0max
        self.n_fft = n_fft
        self.n_shift = n_shift
        self.mcep_dim = mcep_dim
        self.mcep_alpha = mcep_alpha

    def compute(
        self,
        gen_path: Union[str, Path],
        gt_path: Union[str, Path],
    ) -> Dict[str, float]:
        """Compute VUV Error for a single audio pair."""
        return compute_vuv_error_pair(
            gen_path, gt_path,
            f0min=self.f0min, f0max=self.f0max,
            n_fft=self.n_fft, n_shift=self.n_shift,
            mcep_dim=self.mcep_dim, mcep_alpha=self.mcep_alpha,
        )

    def compute_batch(
        self,
        gen_paths: List[Union[str, Path]],
        gt_paths: List[Union[str, Path]],
    ) -> Dict[str, object]:
        """Compute VUV Error for multiple audio pairs."""
        assert len(gen_paths) == len(gt_paths)

        results = {}
        for gen_path, gt_path in zip(gen_paths, gt_paths):
            basename = Path(gen_path).stem
            r = compute_vuv_error_pair(
                gen_path, gt_path,
                f0min=self.f0min, f0max=self.f0max,
                n_fft=self.n_fft, n_shift=self.n_shift,
                mcep_dim=self.mcep_dim, mcep_alpha=self.mcep_alpha,
            )
            results[basename] = r["vuv_error_pct"]

        values = list(results.values())
        return {
            "mean_vuv_error_pct": round(float(np.mean(values)), 2),
            "std_vuv_error_pct": round(float(np.std(values)), 2),
            "num_utterances": len(values),
            "details": results,
        }

    def compute_from_dir(
        self,
        gen_dir: Union[str, Path],
        gt_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, object]:
        """Compute VUV Error for all matching files between two directories."""
        gen_files = find_audio_files(str(gen_dir))
        gt_files = find_audio_files(str(gt_dir))

        if not gen_files:
            raise FileNotFoundError(f"No audio files found in {gen_dir}")
        if not gt_files:
            raise FileNotFoundError(f"No audio files found in {gt_dir}")

        gt_map = {Path(f).stem: f for f in gt_files}
        matched_gen, matched_gt = [], []
        for gen_path in gen_files:
            basename = Path(gen_path).stem
            if basename in gt_map:
                matched_gen.append(gen_path)
                matched_gt.append(gt_map[basename])

        if not matched_gen:
            raise ValueError("No matching files between gen_dir and gt_dir.")

        logging.info(f"Found {len(matched_gen)} matched utterances")
        result = self.compute_batch(matched_gen, matched_gt)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_dir / "utt2vuv_error", "w") as f:
                for utt_id in sorted(result["details"].keys()):
                    f.write(f"{utt_id} {result['details'][utt_id]:.2f}%\n")

            with open(output_dir / "vuv_error_avg_result.txt", "w") as f:
                f.write(f"#utterances: {result['num_utterances']}\n")
                f.write(f"Average: {result['mean_vuv_error_pct']:.2f}%\n")

            result["output_dir"] = str(output_dir)

        return result
