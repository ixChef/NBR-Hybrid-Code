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


def buildAlphaGrid(alphaStep: float) -> list[float]:
    if alphaStep <= 0 or alphaStep > 1:
        raise ValueError("alpha-step must be in the interval (0, 1]")
    alphaCount = int(round(1.0 / alphaStep))
    return [round(index * alphaStep, 10) for index in range(alphaCount + 1)]


def metricSummary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def runSingleSeedFusion(
    leftDir: Path,
    rightDir: Path,
    topk: int,
    alphaStep: float,
) -> dict:
    alignmentSummary = assertAlignedExports(leftDir, rightDir)

    leftDev = loadScoreBundle(leftDir, "dev")
    rightDev = loadScoreBundle(rightDir, "dev")
    leftTest = loadScoreBundle(leftDir, "test")
    rightTest = loadScoreBundle(rightDir, "test")

    leftDevNorm = minMaxNormalisePerUser(leftDev["scores"])
    rightDevNorm = minMaxNormalisePerUser(rightDev["scores"])
    leftTestNorm = minMaxNormalisePerUser(leftTest["scores"])
    rightTestNorm = minMaxNormalisePerUser(rightTest["scores"])

    alphaGrid = buildAlphaGrid(alphaStep)

    bestAlpha = None
    bestDevMetrics = None
    bestDevScore = None
    devGridResults = []

    for alpha in alphaGrid:
        fusedDev = alpha * leftDevNorm + (1.0 - alpha) * rightDevNorm
        devMetrics = evaluateScoreMatrix(fusedDev, leftDev["targets"], topk)
        row = {
            "alpha": float(alpha),
            "precision": float(devMetrics["precision"]),
            "recall": float(devMetrics["recall"]),
            "ndcg": float(devMetrics["ndcg"]),
        }
        devGridResults.append(row)

        scoreTuple = (row["ndcg"], row["recall"], row["precision"])
        if bestDevScore is None or scoreTuple > bestDevScore:
            bestDevScore = scoreTuple
            bestAlpha = float(alpha)
            bestDevMetrics = row

    fusedTest = bestAlpha * leftTestNorm + (1.0 - bestAlpha) * rightTestNorm
    testMetrics = evaluateScoreMatrix(fusedTest, leftTest["targets"], topk)

    return {
        "bestAlpha": float(bestAlpha),
        "bestDevMetrics": bestDevMetrics,
        "testMetrics": {
            "precision": float(testMetrics["precision"]),
            "recall": float(testMetrics["recall"]),
            "ndcg": float(testMetrics["ndcg"]),
        },
        "alignmentSummary": alignmentSummary,
        "alphaGridResults": devGridResults,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-root", required=True)
    parser.add_argument("--right-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    leftRoot = Path(args.left_root).resolve()
    rightDir = Path(args.right_dir).resolve()
    outputDir = Path(args.output_dir).resolve()
    outputDir.mkdir(parents=True, exist_ok=True)

    rightMeta = readJson(rightDir / "meta.json")

    perSeedResults = []
    precisionValues = []
    recallValues = []
    ndcgValues = []
    alphaValues = []

    for seed in args.seeds:
        leftDir = leftRoot / f"seed_{int(seed)}"
        if not leftDir.exists():
            raise FileNotFoundError(f"Missing left export directory: {leftDir}")

        leftMeta = readJson(leftDir / "meta.json")
        result = runSingleSeedFusion(
            leftDir=leftDir,
            rightDir=rightDir,
            topk=int(args.topk),
            alphaStep=float(args.alpha_step),
        )

        row = {
            "seed": int(seed),
            "leftDir": str(leftDir),
            "leftMeta": leftMeta,
            "bestAlpha": float(result["bestAlpha"]),
            "bestDevMetrics": result["bestDevMetrics"],
            "testMetrics": result["testMetrics"],
            "alignmentSummary": result["alignmentSummary"],
            "alphaGridResults": result["alphaGridResults"],
        }
        perSeedResults.append(row)

        precisionValues.append(float(result["testMetrics"]["precision"]))
        recallValues.append(float(result["testMetrics"]["recall"]))
        ndcgValues.append(float(result["testMetrics"]["ndcg"]))
        alphaValues.append(float(result["bestAlpha"]))

    summary = {
        "dataset": rightMeta["dataset"],
        "topk": int(args.topk),
        "alphaStep": float(args.alpha_step),
        "seeds": [int(seed) for seed in args.seeds],
        "nSeeds": len(args.seeds),
        "rightDir": str(rightDir),
        "rightMeta": rightMeta,
        "precision": metricSummary(precisionValues),
        "recall": metricSummary(recallValues),
        "ndcg": metricSummary(ndcgValues),
        "bestAlpha": metricSummary(alphaValues),
        "perSeedBestAlpha": alphaValues,
    }

    writeJson(outputDir / "fusion_per_seed.json", {"results": perSeedResults})
    writeJson(outputDir / "fusion_summary.json", summary)

    print(f"Saved per-seed results to: {outputDir / 'fusion_per_seed.json'}")
    print(f"Saved summary to: {outputDir / 'fusion_summary.json'}")
    print(f"Precision@{args.topk}: {summary['precision']}")
    print(f"Recall@{args.topk}: {summary['recall']}")
    print(f"NDCG@{args.topk}: {summary['ndcg']}")
    print(f"Best alpha summary: {summary['bestAlpha']}")


if __name__ == "__main__":
    main()