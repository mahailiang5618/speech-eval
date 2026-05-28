"""Log-F0 RMSE metric for TTS evaluation.

Inspired by ESPnet's evaluate_f0.py, measures pitch accuracy by computing
the Root Mean Square Error of log-F0 between generated and ground truth audio.
Uses World vocoder for F0 extraction and DTW for temporal alignment.
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
            f"Missing dependencies for F0 RMSE: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
        )


def _get_best_mcep_params(fs: int) -> Tuple[int, float]:
    """Get best mcep dimension and alpha for a given sampling rate."""
    params = {
        16000: (23, 0.42),
        22050: (34, 0.45),
        24000: (34, 0.46),
        44100: (39, 0.53),
        48000: (39, 0.55),
    }
    if fs not in params:
        raise ValueError(
            f"No preset mcep params for fs={fs}. "
            f"Supported: {list(params.keys())}. "
            f"Please specify mcep_dim and mcep_alpha manually."
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
    """Extract World-based acoustic features (mel-cepstrum and F0).

    Args:
        x: 1D waveform array.
        fs: Sampling rate.
        f0min: Minimum F0 value in Hz.
        f0max: Maximum F0 value in Hz.
        n_fft: FFT length in points.
        n_shift: Shift length in points.
        mcep_dim: Dimension of mel-cepstrum (auto-set if None).
        mcep_alpha: All-pass filter coefficient (auto-set if None).

    Returns:
        Tuple of (mel-cepstrum array, F0 array).
    """
    _check_dependencies()

    x = x.astype(np.float64)
    f0, time_axis = pw.harvest(
        x,
        fs,
        f0_floor=f0min,
        f0_ceil=f0max,
        frame_period=n_shift / fs * 1000,
    )
    sp = pw.cheaptrick(x, f0, time_axis, fs, fft_size=n_fft)

    if mcep_dim is None or mcep_alpha is None:
        mcep_dim, mcep_alpha = _get_best_mcep_params(fs)
    mcep = pysptk.sp2mc(sp, mcep_dim, mcep_alpha)

    return mcep, f0


def compute_f0_rmse_pair(
    gen_path: Union[str, Path],
    gt_path: Union[str, Path],
    f0min: int = 40,
    f0max: int = 800,
    n_fft: int = 1024,
    n_shift: int = 256,
    mcep_dim: Optional[int] = None,
    mcep_alpha: Optional[float] = None,
) -> float:
    """Compute log-F0 RMSE between a single pair of audio files.

    Args:
        gen_path: Path to generated/synthesized audio.
        gt_path: Path to ground truth audio.
        f0min: Minimum F0 in Hz.
        f0max: Maximum F0 in Hz.
        n_fft: FFT length.
        n_shift: Shift length.
        mcep_dim: Mel-cepstrum dimension (auto if None).
        mcep_alpha: All-pass coefficient (auto if None).

    Returns:
        Log-F0 RMSE value (lower is better).
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

    # DTW alignment using mcep
    _, path = fastdtw(gen_mcep, gt_mcep, dist=spatial.distance.euclidean)
    twf = np.array(path).T
    gen_f0_dtw = gen_f0[twf[0]]
    gt_f0_dtw = gt_f0[twf[1]]

    # Extract voiced segments (both non-zero)
    nonzero_idxs = np.where((gen_f0_dtw != 0) & (gt_f0_dtw != 0))[0]
    if len(nonzero_idxs) == 0:
        logging.warning("No voiced frames found in both signals.")
        return float("nan")

    gen_f0_voiced = np.log(gen_f0_dtw[nonzero_idxs])
    gt_f0_voiced = np.log(gt_f0_dtw[nonzero_idxs])

    log_f0_rmse = np.sqrt(np.mean((gen_f0_voiced - gt_f0_voiced) ** 2))
    return float(log_f0_rmse)


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


class F0RMSE:
    """Compute Log-F0 RMSE for TTS evaluation.

    Measures pitch accuracy between generated and ground truth speech
    by computing RMSE on log-F0 of voiced segments after DTW alignment.

    Args:
        f0min: Minimum F0 value in Hz for extraction.
        f0max: Maximum F0 value in Hz for extraction.
        n_fft: FFT length in points.
        n_shift: Shift length in points.
        mcep_dim: Mel-cepstrum dimension for DTW (auto if None).
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
        """Compute log-F0 RMSE for a single audio pair.

        Args:
            gen_path: Path to generated audio file.
            gt_path: Path to ground truth audio file.

        Returns:
            Dict with "log_f0_rmse" value.
        """
        rmse = compute_f0_rmse_pair(
            gen_path, gt_path,
            f0min=self.f0min, f0max=self.f0max,
            n_fft=self.n_fft, n_shift=self.n_shift,
            mcep_dim=self.mcep_dim, mcep_alpha=self.mcep_alpha,
        )
        return {"log_f0_rmse": round(rmse, 4)}

    def compute_batch(
        self,
        gen_paths: List[Union[str, Path]],
        gt_paths: List[Union[str, Path]],
    ) -> Dict[str, object]:
        """Compute log-F0 RMSE for multiple audio pairs.

        Args:
            gen_paths: List of generated audio file paths.
            gt_paths: List of ground truth audio file paths.

        Returns:
            Dict with "mean_log_f0_rmse", "std_log_f0_rmse", and per-file "details".
        """
        assert len(gen_paths) == len(gt_paths), (
            f"Mismatch: {len(gen_paths)} generated vs {len(gt_paths)} ground truth files"
        )

        results = {}
        for gen_path, gt_path in zip(gen_paths, gt_paths):
            basename = Path(gen_path).stem
            rmse = compute_f0_rmse_pair(
                gen_path, gt_path,
                f0min=self.f0min, f0max=self.f0max,
                n_fft=self.n_fft, n_shift=self.n_shift,
                mcep_dim=self.mcep_dim, mcep_alpha=self.mcep_alpha,
            )
            results[basename] = round(rmse, 4)

        values = [v for v in results.values() if not np.isnan(v)]
        return {
            "mean_log_f0_rmse": round(float(np.mean(values)), 4),
            "std_log_f0_rmse": round(float(np.std(values)), 4),
            "num_utterances": len(values),
            "details": results,
        }

    def compute_from_dir(
        self,
        gen_dir: Union[str, Path],
        gt_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, object]:
        """Compute log-F0 RMSE for all matching audio files between two directories.

        Files are matched by basename (filename without extension).

        Args:
            gen_dir: Directory of generated audio files.
            gt_dir: Directory of ground truth audio files.
            output_dir: If provided, write results to this directory.

        Returns:
            Dict with mean/std log-F0 RMSE and per-utterance details.
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

            with open(output_dir / "utt2log_f0_rmse", "w") as f:
                for utt_id in sorted(result["details"].keys()):
                    f.write(f"{utt_id} {result['details'][utt_id]:.4f}\n")

            with open(output_dir / "log_f0_rmse_avg_result.txt", "w") as f:
                f.write(f"#utterances: {result['num_utterances']}\n")
                f.write(f"Average: {result['mean_log_f0_rmse']:.4f} +/- {result['std_log_f0_rmse']:.4f}\n")

            result["output_dir"] = str(output_dir)

        return result
