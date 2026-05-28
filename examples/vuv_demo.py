"""VUV Error demo — voiced/unvoiced classification error rate."""

from speech_eval.common import VUVError, compute_vuv_error_pair


def main():
    gen_wav = "path/to/generated.wav"
    gt_wav = "path/to/ground_truth.wav"

    # --- Method 1: Quick function ---
    result = compute_vuv_error_pair(gen_wav, gt_wav)
    print("=== Method 1: Quick function ===")
    print(f"VUV Error: {result['vuv_error_pct']}%")
    print(f"Error frames: {result['error_frames']} / {result['total_frames']}")

    # --- Method 2: Class interface ---
    metric = VUVError(f0min=40, f0max=800)

    print("\n=== Method 2: Single pair ===")
    result = metric.compute(gen_wav, gt_wav)
    print(f"VUV Error: {result['vuv_error_pct']}%")

    # --- Method 3: Batch ---
    print("\n=== Method 3: Batch ===")
    gen_paths = ["gen_001.wav", "gen_002.wav", "gen_003.wav"]
    gt_paths = ["gt_001.wav", "gt_002.wav", "gt_003.wav"]
    result = metric.compute_batch(gen_paths, gt_paths)
    print(f"Mean VUV Error: {result['mean_vuv_error_pct']}%")
    print(f"Std: {result['std_vuv_error_pct']}%")
    print(f"Utterances: {result['num_utterances']}")

    # --- Method 4: Directory ---
    print("\n=== Method 4: Directory ===")
    result = metric.compute_from_dir(
        gen_dir="output/generated/",
        gt_dir="data/ground_truth/",
        output_dir="results/vuv/",
    )
    print(f"Mean VUV Error: {result['mean_vuv_error_pct']}%")
    print(f"Results saved to: {result.get('output_dir', 'N/A')}")


if __name__ == "__main__":
    main()
