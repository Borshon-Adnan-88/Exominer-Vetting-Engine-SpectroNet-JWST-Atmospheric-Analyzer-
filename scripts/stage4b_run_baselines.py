"""Run approved Stage 4B baselines or replay predictions without refitting."""
import argparse
import os

# Set thread policy before importing numerical libraries.
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

from kepler_baselines.runner import run, verify_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--output", default="data/stage4b_runtime/baseline_v1")
    parser.add_argument("--metadata", default="data/stage4b")
    parser.add_argument("--config", default="configs/kepler_stage4b_baselines_v1.json")
    parser.add_argument("--verify-predictions", action="store_true")
    args = parser.parse_args()
    if args.verify_predictions:
        verify_predictions(args.output, args.metadata, args.config)
    else:
        if not args.dataset:
            parser.error("--dataset is required for the fitting run")
        run(args.dataset, args.output, args.metadata, args.config)


if __name__ == "__main__":
    main()
