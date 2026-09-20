"""Sequential approved CNN fits, or fresh-process replay with no fitting."""
import argparse
import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    from kepler_cnn.runner import run
    if not args.replay and not args.dataset:
        parser.error("--dataset required for canonical validation before fitting")
    run(args.dataset, replay=args.replay)


if __name__ == "__main__":
    main()
