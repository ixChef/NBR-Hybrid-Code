from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def getScoreExportRoot(projectRoot: Path) -> Path:
    outputRoot = projectRoot / "results" / "score_exports"
    outputRoot.mkdir(parents=True, exist_ok=True)
    return outputRoot


def writeJson(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def readJson(path: Path) -> dict:
    return json.loads(path.read_text())


def normaliseTarget(rawTarget) -> list[int]:
    if isinstance(rawTarget, np.ndarray):
        values = rawTarget.reshape(-1).tolist()
    elif hasattr(rawTarget, "detach"):
        values = rawTarget.detach().cpu().reshape(-1).tolist()
    elif isinstance(rawTarget, (list, tuple, set)):
        values = list(rawTarget)
    else:
        values = [rawTarget]

    normalised = []
    for value in values:
        if isinstance(value, np.ndarray):
            normalised.extend(int(x) for x in value.reshape(-1).tolist())
        elif hasattr(value, "detach"):
            normalised.extend(int(x) for x in value.detach().cpu().reshape(-1).tolist())
        elif isinstance(value, (list, tuple, set)):
            normalised.extend(int(x) for x in list(value))
        else:
            normalised.append(int(value))

    return normalised


def saveTargets(path: Path, targets: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = [[int(x) for x in target] for target in targets]
    path.write_text(json.dumps(serialisable))


def loadTargets(path: Path) -> list[list[int]]:
    payload = json.loads(path.read_text())
    return [[int(x) for x in target] for target in payload]


def saveScoreBundle(
    outputDir: Path,
    splitName: str,
    scores: np.ndarray,
    userIds: np.ndarray,
    targets: list[list[int]],
) -> None:
    outputDir.mkdir(parents=True, exist_ok=True)
    np.save(outputDir / f"scores_{splitName}.npy", scores.astype(np.float32, copy=False))
    np.save(outputDir / f"user_ids_{splitName}.npy", userIds.astype(np.int64, copy=False))
    saveTargets(outputDir / f"targets_{splitName}.json", targets)


def saveItemIds(outputDir: Path, itemIds: np.ndarray) -> None:
    outputDir.mkdir(parents=True, exist_ok=True)
    np.save(outputDir / "item_ids.npy", itemIds.astype(np.int64, copy=False))


def loadScoreBundle(outputDir: Path, splitName: str) -> dict:
    return {
        "scores": np.load(outputDir / f"scores_{splitName}.npy"),
        "userIds": np.load(outputDir / f"user_ids_{splitName}.npy"),
        "targets": loadTargets(outputDir / f"targets_{splitName}.json"),
    }


def assertAlignedExports(leftDir: Path, rightDir: Path) -> dict:
    leftMeta = readJson(leftDir / "meta.json")
    rightMeta = readJson(rightDir / "meta.json")

    leftItemIds = np.load(leftDir / "item_ids.npy")
    rightItemIds = np.load(rightDir / "item_ids.npy")

    if leftItemIds.shape != rightItemIds.shape or not np.array_equal(leftItemIds, rightItemIds):
        raise ValueError("item_ids are not aligned between the two exports")

    summary = {
        "leftDir": str(leftDir),
        "rightDir": str(rightDir),
        "itemCount": int(leftItemIds.shape[0]),
        "datasetLeft": leftMeta.get("dataset"),
        "datasetRight": rightMeta.get("dataset"),
        "splitChecks": {},
    }

    for splitName in ["dev", "test"]:
        leftBundle = loadScoreBundle(leftDir, splitName)
        rightBundle = loadScoreBundle(rightDir, splitName)

        if leftBundle["scores"].shape != rightBundle["scores"].shape:
            raise ValueError(f"{splitName} score matrices do not have the same shape")

        if not np.array_equal(leftBundle["userIds"], rightBundle["userIds"]):
            raise ValueError(f"{splitName} user_ids are not aligned")

        if leftBundle["targets"] != rightBundle["targets"]:
            raise ValueError(f"{splitName} targets are not aligned")

        summary["splitChecks"][splitName] = {
            "rows": int(leftBundle["scores"].shape[0]),
            "cols": int(leftBundle["scores"].shape[1]),
        }

    return summary


def minMaxNormalisePerUser(scores: np.ndarray) -> np.ndarray:
    scores = scores.astype(np.float32, copy=False)
    minValues = scores.min(axis=1, keepdims=True)
    maxValues = scores.max(axis=1, keepdims=True)
    ranges = maxValues - minValues
    safeRanges = np.where(ranges == 0.0, 1.0, ranges)
    normalised = (scores - minValues) / safeRanges
    normalised[ranges.reshape(-1) == 0.0] = 0.0
    return normalised.astype(np.float32, copy=False)


def precisionAtK(rankedItems: np.ndarray, targetItems: list[int], k: int) -> float:
    targetSet = set(int(x) for x in targetItems)
    if k <= 0:
        return 0.0
    hits = sum(1 for item in rankedItems[:k] if int(item) in targetSet)
    return float(hits) / float(k)


def recallAtK(rankedItems: np.ndarray, targetItems: list[int], k: int) -> float:
    targetSet = set(int(x) for x in targetItems)
    if len(targetSet) == 0:
        return 0.0
    hits = sum(1 for item in rankedItems[:k] if int(item) in targetSet)
    return float(hits) / float(len(targetSet))


def ndcgAtK(rankedItems: np.ndarray, targetItems: list[int], k: int) -> float:
    targetSet = set(int(x) for x in targetItems)
    if len(targetSet) == 0:
        return 0.0

    dcg = 0.0
    for rank, item in enumerate(rankedItems[:k], start=1):
        if int(item) in targetSet:
            dcg += 1.0 / np.log2(rank + 1.0)

    idealHits = min(len(targetSet), k)
    if idealHits == 0:
        return 0.0

    idcg = sum(1.0 / np.log2(rank + 1.0) for rank in range(1, idealHits + 1))
    return float(dcg / idcg)


def evaluateScoreMatrix(scores: np.ndarray, targets: list[list[int]], topk: int) -> dict:
    precisions = []
    recalls = []
    ndcgs = []

    for rowIndex in range(scores.shape[0]):
        rankedItems = np.argsort(-scores[rowIndex])[:topk]
        targetItems = targets[rowIndex]
        precisions.append(precisionAtK(rankedItems, targetItems, topk))
        recalls.append(recallAtK(rankedItems, targetItems, topk))
        ndcgs.append(ndcgAtK(rankedItems, targetItems, topk))

    return {
        "precision": float(np.mean(np.asarray(precisions, dtype=np.float64))),
        "recall": float(np.mean(np.asarray(recalls, dtype=np.float64))),
        "ndcg": float(np.mean(np.asarray(ndcgs, dtype=np.float64))),
    }