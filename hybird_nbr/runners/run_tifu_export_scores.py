from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

projectRoot = Path(__file__).resolve().parents[1]
workspaceRoot = projectRoot.parent
tifuRepoRoot = workspaceRoot / "time_dependent_nbr"

if str(projectRoot) not in sys.path:
    sys.path.insert(0, str(projectRoot))
if str(tifuRepoRoot) not in sys.path:
    sys.path.insert(0, str(tifuRepoRoot))

import src.dataset.base as tifuBaseModule
import src.settings as tifuSettings
from src.dataset import DATASETS
from src.models import IRecommenderNextTs, MODELS
from src.utils import set_global_seed

from runners.fusion_common import (
    getScoreExportRoot,
    normaliseTarget,
    saveItemIds,
    saveScoreBundle,
    writeJson,
)
from runners.run_tifu_on_profile import ensureAlias, resolveModelKey


def getItemCount(profileRoot: Path, aliasName: str) -> int:
    itemMappingPath = profileRoot / aliasName / "split" / "item_mapping.csv"
    if not itemMappingPath.exists():
        raise FileNotFoundError(f"Missing item_mapping.csv at {itemMappingPath}")
    itemMappingDf = pd.read_csv(itemMappingPath)
    return int(itemMappingDf.shape[0])


def getStudy(prefix: str):
    resultsDir = Path(tifuSettings.RESULTS_DIR)
    studyDbPath = resultsDir / f"{prefix}.db"
    if not studyDbPath.exists():
        raise FileNotFoundError(f"Missing TIFU Optuna DB at {studyDbPath}")
    storageUrl = f"sqlite:///{studyDbPath.as_posix()}"
    return optuna.load_study(study_name=prefix, storage=storageUrl)


def collectTifuScores(
    model,
    datasetDf: pd.DataFrame,
    itemCount: int,
    batchSize: int,
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    scoreRows = []
    userIds = []
    targets = []

    totalRows = int(datasetDf.shape[0])
    batchStart = 0

    while batchStart < totalRows:
        batchEnd = min(batchStart + batchSize, totalRows)
        batchDf = datasetDf.iloc[batchStart:batchEnd]

        usersBatch = batchDf.user_id
        if isinstance(model, IRecommenderNextTs):
            nextBasketDf = batchDf.loc[:, ["user_id", "timestamp"]].rename(columns={"timestamp": "next_basket_ts"})
            scoresBatch = model.predict(
                usersBatch.to_numpy(),
                nextBasketDf,
                topk=itemCount,
            )
        else:
            scoresBatch = model.predict(
                usersBatch.to_numpy(),
                topk=itemCount,
            )

        if hasattr(scoresBatch, "toarray"):
            denseBatch = scoresBatch.toarray()
        else:
            denseBatch = np.asarray(scoresBatch)

        if denseBatch.ndim == 1:
            denseBatch = denseBatch.reshape(1, -1)

        if denseBatch.shape[1] != itemCount:
            raise ValueError(
                f"Expected TIFU scores with {itemCount} columns, got {denseBatch.shape[1]} columns."
            )

        denseBatch = denseBatch.astype(np.float32, copy=False)

        for rowIndex in range(denseBatch.shape[0]):
            scoreRows.append(denseBatch[rowIndex])
            userIds.append(int(batchDf.iloc[rowIndex].user_id))
            targets.append(normaliseTarget(batchDf.iloc[rowIndex].basket))

        batchStart = batchEnd

    scores = np.vstack(scoreRows).astype(np.float32, copy=False)
    userIdsArray = np.asarray(userIds, dtype=np.int64)
    return scores, userIdsArray, targets


def buildOutputDir(args: argparse.Namespace, resolvedModel: str) -> Path:
    exportRoot = getScoreExportRoot(projectRoot)
    return exportRoot / "tifu" / args.alias_prefix / args.dataset / resolvedModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["dunnhumby", "tafeng"])
    parser.add_argument("--model", default="tifuknn_time_days_next_ts")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--profile-root",
        default=str(projectRoot / "data" / "taiw_profile_tifu"),
    )
    parser.add_argument("--alias-prefix", default="taiw_profile")
    args = parser.parse_args()

    profileRoot = Path(args.profile_root)
    aliasName = f"{args.alias_prefix}_{args.dataset}"
    ensureAlias(profileRoot, args.dataset, aliasName)

    resolvedModel = resolveModelKey(args.model)

    tifuSettings.DATA_DIR = str(profileRoot)
    tifuBaseModule.DATA_DIR = str(profileRoot)

    datasetCls = DATASETS[args.dataset]
    modelCls = MODELS[resolvedModel]

    data = datasetCls(aliasName, verbose=True)
    data.load_split()

    prefix = f"{aliasName}_{resolvedModel}"
    study = getStudy(prefix)

    set_global_seed(42)
    bestModel = modelCls(**study.best_params)

    set_global_seed(42)
    bestModel.fit(dataset=data)

    itemCount = getItemCount(profileRoot, aliasName)
    outputDir = buildOutputDir(args, resolvedModel)

    scoresDev, userIdsDev, targetsDev = collectTifuScores(
        model=bestModel,
        datasetDf=data.val_df,
        itemCount=itemCount,
        batchSize=int(args.batch_size),
    )
    scoresTest, userIdsTest, targetsTest = collectTifuScores(
        model=bestModel,
        datasetDf=data.test_df,
        itemCount=itemCount,
        batchSize=int(args.batch_size),
    )

    saveItemIds(outputDir, np.arange(itemCount, dtype=np.int64))
    saveScoreBundle(outputDir, "dev", scoresDev, userIdsDev, targetsDev)
    saveScoreBundle(outputDir, "test", scoresTest, userIdsTest, targetsTest)

    meta = {
        "modelFamily": "TIFU",
        "dataset": args.dataset,
        "profileRoot": str(profileRoot),
        "aliasPrefix": args.alias_prefix,
        "aliasName": aliasName,
        "resolvedModel": resolvedModel,
        "studyName": prefix,
        "bestTrialNumber": int(study.best_trial.number),
        "bestValue": float(study.best_value),
        "bestParams": dict(study.best_params),
        "predictTopkArg": int(itemCount),
        "itemCount": int(itemCount),
        "devRows": int(scoresDev.shape[0]),
        "testRows": int(scoresTest.shape[0]),
        "outputDir": str(outputDir),
    }
    writeJson(outputDir / "meta.json", meta)

    print(f"Saved TIFU export to: {outputDir}")


if __name__ == "__main__":
    main()