"""Character Error Rate (CER) metric for ASR evaluation.

Inspired by ESPnet3's CER implementation, provides both a class-based interface
and a simple function interface for computing CER. Particularly suited for
Chinese and other languages without explicit word boundaries.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    import jiwer
except ImportError:
    jiwer = None


def _check_jiwer():
    if jiwer is None:
        raise RuntimeError(
            "jiwer is required to compute CER. "
            "Install it with: pip install jiwer"
        )


def _normalize_text(text: str, lowercase: bool = True, remove_punctuation: bool = True) -> str:
    """Normalize text for CER computation."""
    if lowercase:
        text = text.lower()
    if remove_punctuation:
        text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "."


def _char_tokenize(text: str) -> str:
    """Insert spaces between every character for character-level comparison."""
    return " ".join(list(text.replace(" ", "")))


class CER:
    """Compute Character Error Rate for ASR hypotheses.

    CER operates at the character level, making it ideal for Chinese, Japanese,
    and other languages where word segmentation is ambiguous.

    Args:
        lowercase: Whether to lowercase text before comparison.
        remove_punctuation: Whether to remove punctuation before comparison.
    """

    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True) -> None:
        _check_jiwer()
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation

    def _clean(self, text: str) -> str:
        return _normalize_text(text, self.lowercase, self.remove_punctuation)

    def compute(
        self,
        reference: Union[str, List[str]],
        hypothesis: Union[str, List[str]],
    ) -> Dict[str, Union[float, int]]:
        """Compute CER between reference and hypothesis.

        Args:
            reference: Reference text(s).
            hypothesis: Hypothesis text(s).

        Returns:
            Dict with keys:
                - "cer": overall CER as a percentage
                - "substitutions": total character substitution count
                - "deletions": total character deletion count
                - "insertions": total character insertion count
                - "ref_char_count": total characters in reference
        """
        if isinstance(reference, str):
            reference = [reference]
            hypothesis = [hypothesis]

        assert len(reference) == len(hypothesis), (
            f"Reference count ({len(reference)}) != Hypothesis count ({len(hypothesis)})"
        )

        refs = [self._clean(r) for r in reference]
        hyps = [self._clean(h) for h in hypothesis]

        output = jiwer.process_characters(refs, hyps)

        return {
            "cer": round(output.cer * 100, 2),
            "substitutions": output.substitutions,
            "deletions": output.deletions,
            "insertions": output.insertions,
            "ref_char_count": sum(len(r.replace(" ", "")) for r in refs),
        }

    def compute_detail(
        self,
        reference: Union[str, List[str]],
        hypothesis: Union[str, List[str]],
    ) -> Dict[str, object]:
        """Compute CER with per-sentence detail and alignment visualization.

        Returns:
            Dict with keys:
                - "cer": overall CER percentage
                - "sentence_cers": per-sentence CER list
                - "alignment": human-readable alignment string
                - "substitutions", "deletions", "insertions": error counts
        """
        if isinstance(reference, str):
            reference = [reference]
            hypothesis = [hypothesis]

        refs = [self._clean(r) for r in reference]
        hyps = [self._clean(h) for h in hypothesis]

        output = jiwer.process_characters(refs, hyps)
        alignment_str = jiwer.visualize_alignment(output, show_measures=True)

        sentence_cers = []
        for ref, hyp in zip(refs, hyps):
            s_cer = jiwer.cer(ref, hyp) * 100
            sentence_cers.append(round(s_cer, 2))

        return {
            "cer": round(output.cer * 100, 2),
            "sentence_cers": sentence_cers,
            "alignment": alignment_str,
            "substitutions": output.substitutions,
            "deletions": output.deletions,
            "insertions": output.insertions,
        }

    def compute_from_file(
        self,
        ref_path: Union[str, Path],
        hyp_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Union[float, str]]:
        """Compute CER from reference and hypothesis text files.

        Each file should contain one sentence per line, aligned by line number.

        Args:
            ref_path: Path to reference text file.
            hyp_path: Path to hypothesis text file.
            output_dir: If provided, write alignment visualization to this directory.

        Returns:
            Dict with CER results. If output_dir is set, includes "alignment_file" path.
        """
        ref_path = Path(ref_path)
        hyp_path = Path(hyp_path)

        refs = ref_path.read_text(encoding="utf-8").strip().splitlines()
        hyps = hyp_path.read_text(encoding="utf-8").strip().splitlines()

        assert len(refs) == len(hyps), (
            f"Line count mismatch: ref has {len(refs)} lines, hyp has {len(hyps)} lines"
        )

        result = self.compute_detail(refs, hyps)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            alignment_file = output_dir / "cer_alignment.txt"
            alignment_file.write_text(result["alignment"], encoding="utf-8")
            result["alignment_file"] = str(alignment_file)

        return result


def compute_cer(
    reference: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    lowercase: bool = True,
    remove_punctuation: bool = True,
) -> float:
    """Quick function to compute CER percentage.

    Args:
        reference: Reference text(s).
        hypothesis: Hypothesis text(s).
        lowercase: Normalize to lowercase.
        remove_punctuation: Remove punctuation before comparison.

    Returns:
        CER as a percentage (e.g., 16.67 means 16.67%).
    """
    _check_jiwer()

    if isinstance(reference, str):
        reference = [reference]
        hypothesis = [hypothesis]

    refs = [_normalize_text(r, lowercase, remove_punctuation) for r in reference]
    hyps = [_normalize_text(h, lowercase, remove_punctuation) for h in hypothesis]

    return round(jiwer.cer(refs, hyps) * 100, 2)
