from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

projectRoot = Path(__file__).resolve().parents[1]
if str(projectRoot) not in sys.path:
    sys.path.insert(0, str(projectRoot))

from runners.fusion_common import (
    assertAlignedExports,
    evaluateScoreMatrix,
    loadScoreBundle,
    minMaxNormalisePerUser,
    readJson,
    writeJson,
)


def buildDefaultOutputDir(leftDir: Path, rightDir: Path) -> Path:
    leftMeta = readJson(leftDir / "meta.json")
    rightMeta = readJson(rightDir / "meta.json")
    dataset = leftMeta["dataset"]
    profileName = leftMeta.get("profileName", leftMeta.get("aliasPrefix", "profile"))
    leftName = leftMeta["modelFamily"].lower()
    rightName = rightMeta["modelFamily"].lower()
    return projectRoot / "results" / "score_fusion" / dataset / profileName / f"{leftName}__plus__{rightName}"


def buildAlphaGrid(alphaStep: float) -> list[float]:
    if alphaStep <= 0 or alphaStep > 1:
        raise ValueError("alpha-step must be in the interval (0, 1]")
    alphaCount = int(round(1.0 / alphaStep))
    return [round(index * alphaStep, 10) for index in range(alphaCount + 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-dir", required=True)
    parser.add_argument("--right-dir", required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-fused-scores", action="store_true")
    args = parser.parse_args()

    leftDir = Path(args.left_dir).resolve()
    rightDir = Path(args.right_dir).resolve()

    alignmentSummary = assertAlignedExports(leftDir, rightDir)

    if args.output_dir is None:
        outputDir = buildDefaultOutputDir(leftDir, rightDir)
    else:
        outputDir = Path(args.output_dir).resolve()
    outputDir.mkdir(parents=True, exist_ok=True)

    leftMeta = readJson(leftDir / "meta.json")
    rightMeta = readJson(rightDir / "meta.json")

    leftDev = loadScoreBundle(leftDir, "dev")
    rightDev = loadScoreBundle(rightDir, "dev")
    leftTest = loadScoreBundle(leftDir, "test")
    rightTest = loadScoreBundle(rightDir, "test")

    leftDevNorm = minMaxNormalisePerUser(leftDev["scores"])
    rightDevNorm = minMaxNormalisePerUser(rightDev["scores"])
    leftTestNorm = minMaxNormalisePerUser(leftTest["scores"])
    rightTestNorm = minMaxNormalisePerUser(rightTest["scores"])

    alphaGrid = buildAlphaGrid(float(args.alpha_step))
    devResults = []

    bestAlpha = None
    bestDevMetrics = None
    bestDevScore = None

    for alpha in alphaGrid:
        fusedDev = alpha * leftDevNorm + (1.0 - alpha) * rightDevNorm
        devMetrics = evaluateScoreMatrix(fusedDev, leftDev["targets"], int(args.topk))
        row = {
            "alpha": float(alpha),
            "precision": float(devMetrics["precision"]),
            "recall": float(devMetrics["recall"]),
            "ndcg": float(devMetrics["ndcg"]),
        }
        devResults.append(row)

        scoreTuple = (row["ndcg"], row["recall"], row["precision"])
        if bestDevScore is None or scoreTuple > bestDevScore:
            bestDevScore = scoreTuple
            bestAlpha = float(alpha)
            bestDevMetrics = row

    fusedTest = bestAlpha * leftTestNorm + (1.0 - bestAlpha) * rightTestNorm
    testMetrics = evaluateScoreMatrix(fusedTest, leftTest["targets"], int(args.topk))

    summary = {
        "dataset": leftMeta["dataset"],
        "topk": int(args.topk),
        "alphaStep": float(args.alpha_step),
        "bestAlpha": float(bestAlpha),
        "bestDevMetrics": bestDevMetrics,
        "testMetrics": {
            "precision": float(testMetrics["precision"]),
            "recall": float(testMetrics["recall"]),
            "ndcg": float(testMetrics["ndcg"]),
        },
        "leftExport": str(leftDir),
        "rightExport": str(rightDir),
        "leftMeta": leftMeta,
        "rightMeta": rightMeta,
        "alignmentSummary": alignmentSummary,
        "alphaGridResults": devResults,
    }

    writeJson(outputDir / "fusion_summary.json", summary)

    if args.save_fused_scores:
        np.save(outputDir / "fused_scores_dev.npy", (bestAlpha * leftDevNorm + (1.0 - bestAlpha) * rightDevNorm).astype(np.float32, copy=False))
        np.save(outputDir / "fused_scores_test.npy", fusedTest.astype(np.float32, copy=False))

    print(f"Saved fusion summary to: {outputDir / 'fusion_summary.json'}")
    print(f"Best alpha: {bestAlpha:.2f}")
    print(f"Dev metrics: {bestDevMetrics}")
    print(f"Test metrics: {summary['testMetrics']}")


if __name__ == "__main__":
    main()