from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

projectRoot = Path(__file__).resolve().parents[1]
workspaceRoot = projectRoot.parent
taiwRepoRoot = workspaceRoot / "time_aware_item_weighting"

if str(projectRoot) not in sys.path:
    sys.path.insert(0, str(projectRoot))
if str(taiwRepoRoot) not in sys.path:
    sys.path.insert(0, str(taiwRepoRoot))

from nbr.model.nbrknn import NBRKNN
from runners.fusion_common import (
    getScoreExportRoot,
    normaliseTarget,
    saveItemIds,
    saveScoreBundle,
    writeJson,
)
from runners.taiw_common import buildCorpus, buildTrainer, runSlrcStage
from runners.taiw_route_a import countCompleteTrials, getBestBundle, getRunDir, loadStudy


def collectTaiwScores(
    corpus,
    userEmb: np.ndarray,
    nearestNeighborsNum: int,
    alpha: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    trainer = buildTrainer(
        corpus=corpus,
        maxEpochs=None,
        topk=10,
        earlyStopNum=None,
    )
    model = NBRKNN(
        item_num=corpus.n_items,
        user_num=corpus.n_users,
        nearest_neighbors_num=int(nearestNeighborsNum),
        alpha=float(alpha),
        user_emb=userEmb.astype("float32", copy=False),
    )
    trainer.init_hyperparams(model=model)

    if mode == "dev":
        dataloader = trainer.dev_dataloader
    elif mode == "test":
        dataloader = trainer.test_dataloader
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    trainer.model.eval()

    scoreRows = []
    userIds = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Export {mode} scores"):
            properItems = batch["proper_items"]
            batch = {key: value.to(trainer.device) for key, value in batch.items() if key != "proper_items"}

            itemsScores = trainer.model.predict_for_user(
                user_id=batch["user_id"][0],
                t=batch["t"],
                length=batch["length"],
                history_time=batch["history_time"],
            )
            itemsScores = itemsScores.detach().cpu().numpy().reshape(-1, corpus.n_items)

            scoreRows.append(itemsScores[0].astype(np.float32, copy=False))
            userIds.append(int(batch["user_id"][0].item()))

            rawTarget = properItems[0] if isinstance(properItems, list) and len(properItems) == 1 else properItems
            targets.append(normaliseTarget(rawTarget))

    scores = np.vstack(scoreRows).astype(np.float32, copy=False)
    userIdsArray = np.asarray(userIds, dtype=np.int64)
    return scores, userIdsArray, targets


def buildOutputDir(args: argparse.Namespace) -> Path:
    exportRoot = getScoreExportRoot(projectRoot)
    return exportRoot / "taiw" / args.search_space / args.profile / args.dataset / f"seed_{int(args.eval_seed)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["dunnhumby", "tafeng"])
    parser.add_argument("--profile", required=True, choices=["tifu_profile", "taiw_profile", "tifu_hybrid_profile"])
    parser.add_argument("--profile-root-override", default=None)
    parser.add_argument("--search-space", default="knn_only", choices=["full", "knn_only"])
    parser.add_argument("--tune-seed", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=0)
    args = parser.parse_args()

    runDir = getRunDir(args.dataset, args.profile, args.search_space)
    if not (runDir / "optuna.db").exists():
        raise RuntimeError(f"No TAIW study found at {runDir / 'optuna.db'}")

    study = loadStudy(
        runDir=runDir,
        dataset=args.dataset,
        profileName=args.profile,
        searchSpace=args.search_space,
        tuneSeed=args.tune_seed,
    )
    if countCompleteTrials(study) == 0:
        raise RuntimeError("The TAIW study exists but has no COMPLETE trials.")

    slrcParams, nbrknnParams = getBestBundle(args.dataset, args.search_space, study.best_params)

    corpus, resolvedPrefix = buildCorpus(
        dataset=args.dataset,
        profileName=args.profile,
        profileRootOverride=args.profile_root_override,
    )

    outputDir = buildOutputDir(args)
    workDir = outputDir / "work"

    slrcResult = runSlrcStage(
        corpus=corpus,
        seed=int(args.eval_seed),
        topk=10,
        slrcParams=slrcParams,
        workDir=workDir,
    )

    scoresDev, userIdsDev, targetsDev = collectTaiwScores(
        corpus=corpus,
        userEmb=slrcResult["devUserEmb"],
        nearestNeighborsNum=int(nbrknnParams["nearest_neighbors_num"]),
        alpha=float(nbrknnParams["alpha"]),
        mode="dev",
    )
    scoresTest, userIdsTest, targetsTest = collectTaiwScores(
        corpus=corpus,
        userEmb=slrcResult["testUserEmb"],
        nearestNeighborsNum=int(nbrknnParams["nearest_neighbors_num"]),
        alpha=float(nbrknnParams["alpha"]),
        mode="test",
    )

    saveItemIds(outputDir, np.arange(corpus.n_items, dtype=np.int64))
    saveScoreBundle(outputDir, "dev", scoresDev, userIdsDev, targetsDev)
    saveScoreBundle(outputDir, "test", scoresTest, userIdsTest, targetsTest)

    meta = {
        "modelFamily": "TAIW",
        "dataset": args.dataset,
        "profileName": args.profile,
        "profileRootOverride": args.profile_root_override,
        "searchSpace": args.search_space,
        "tuneSeed": int(args.tune_seed),
        "evalSeed": int(args.eval_seed),
        "studyName": study.study_name,
        "bestTrialNumber": int(study.best_trial.number),
        "bestValue": float(study.best_value),
        "bestParams": dict(study.best_params),
        "slrcParams": slrcParams,
        "nbrknnParams": nbrknnParams,
        "resolvedCorpusPrefix": str(resolvedPrefix),
        "itemCount": int(corpus.n_items),
        "devRows": int(scoresDev.shape[0]),
        "testRows": int(scoresTest.shape[0]),
        "outputDir": str(outputDir),
    }
    writeJson(outputDir / "meta.json", meta)

    print(f"Saved TAIW export to: {outputDir}")


if __name__ == "__main__":
    main()