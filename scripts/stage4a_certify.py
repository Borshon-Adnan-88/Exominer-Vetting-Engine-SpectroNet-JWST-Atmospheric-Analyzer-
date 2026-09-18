"""Finalize already constructed Stage 4A outputs without archive access."""
import argparse
from kepler_scale.finalize import certify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scratch-dataset", required=True)
    args = parser.parse_args()
    certify(args.dataset, args.scratch_dataset)


if __name__ == "__main__":
    main()
