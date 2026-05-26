from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

projectRoot = Path(__file__).resolve().parents[1]
if str(projectRoot) not in sys.path:
    sys.path.insert(0, str(projectRoot))

from runners.fusion_common import assertAlignedExports, writeJson


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-dir", required=True)
    parser.add_argument("--right-dir", required=True)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    leftDir = Path(args.left_dir).resolve()
    rightDir = Path(args.right_dir).resolve()

    summary = assertAlignedExports(leftDir, rightDir)
    summary["isAligned"] = True

    print(json.dumps(summary, indent=2))

    if args.output_json:
        writeJson(Path(args.output_json), summary)


if __name__ == "__main__":
    main()