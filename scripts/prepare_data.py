"""Validate the workbook and write a local dataset with reproducible split manifests."""

import argparse
import json
import tomllib
from pathlib import Path

from amesvalue.data import read_data, split_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/AmesHousing.xls"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    frame, audit = read_data(args.input)
    seed = tomllib.loads(Path("configs/training.toml").read_text())["seed"]
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "validated.csv", index=False)
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2))
    for temporal in [False, True]:
        split = "chronological" if temporal else "random"
        for name, partition in split_data(frame, seed, temporal).items():
            partition[["original_row"]].to_csv(
                args.output / f"{split}_{name}.csv", index=False
            )


if __name__ == "__main__":
    main()
