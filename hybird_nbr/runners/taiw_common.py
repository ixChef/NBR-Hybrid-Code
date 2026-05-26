from __future__ import annotations

import contextlib
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

projectRoot = Path(__file__).resolve().parents[1]
workspaceRoot = projectRoot.parent
taiwRepoRoot = workspaceRoot / "time_aware_item_weighting"

if str(projectRoot) not in sys.path:
    sys.path.insert(0, str(projectRoot))
if str(taiwRepoRoot) not in sys.path:
    sys.path.insert(0, str(taiwRepoRoot))

from configs.profiles import getProfile
from nbr.model.bpr import BPR
from nbr.model.nbrknn import NBRKNN
from nbr.model.slrc import SLRC
from nbr.preparation.corpus import Corpus
from nbr.trainer.trainer import NBRTrainer


def setSeed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def getProfileRoot(profileName: str) -> Path:
    profile = getProfile(profileName)
    return Path(profile["outputRoot"])


def resolveCorpusPrefix(profileName: str, dataset: str, profileRootOverride: str | None = None) -> Path:
    profileRoot = getProfileRoot(profileName)

    candidateRoots = []
    if profileRootOverride:
        candidateRoots.append(Path(profileRootOverride))
    candidateRoots.append(profileRoot)
    candidateRoots.append(projectRoot / "data" / profileName)

    seen = set()
    checked = []

    for root in candidateRoots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)

        directExpected = root / f"data_{dataset}" / "book.csv"
        checked.append(directExpected)
        if directExpected.exists():
            return root

        if root.name == f"data_{dataset}" and (root / "book.csv").exists():
            return root.parent
        checked.append(root / "book.csv")

    checkedText = "\n".join(str(path) for path in checked)
    raise FileNotFoundError(
        f"Could not locate TAIW corpus for profile={profileName}, dataset={dataset}.\n"
        f"Expected a book.csv under one of these layouts:\n{checkedText}\n"
        f"If your generated data lives elsewhere, pass --profile-root-override."
    )


def buildCorpus(dataset: str, profileName: str, profileRootOverride: str | None = None) -> tuple[Corpus, Path]:
    resolvedPrefix = resolveCorpusPrefix(
        profileName=profileName,
        dataset=dataset,
        profileRootOverride=profileRootOverride,
    )
    corpus = Corpus(str(resolvedPrefix), dataset)
    corpus.load_data()
    return corpus, resolvedPrefix


def buildSlrcModel(corpus: Corpus, embSize: int) -> SLRC:
    return SLRC(
        base_model_class=BPR,
        base_model_config={
            "emb_size": embSize,
            "user_num": corpus.n_users,
            "item_num": corpus.n_items,
            "click_num": corpus.n_clicks,
        },
        item_num=corpus.n_items,
        avg_repeat_interval=corpus.total_avg_interval,
    )


def buildTrainer(corpus: Corpus, maxEpochs: int | None, topk: int, earlyStopNum: int | None) -> NBRTrainer:
    return NBRTrainer(
        corpus=corpus,
        max_epochs=maxEpochs,
        topk=topk,
        early_stop_num=earlyStopNum,
    )


@contextlib.contextmanager
def workingDirectory(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def toFloatMetrics(metrics: dict) -> dict:
    return {key: float(value) for key, value in metrics.items()}


def runSlrcStage(
    corpus: Corpus,
    seed: int,
    topk: int,
    slrcParams: dict,
    workDir: Path,
) -> dict:
    setSeed(seed)
    with workingDirectory(workDir):
        trainer = buildTrainer(
            corpus=corpus,
            maxEpochs=int(slrcParams["max_epochs"]),
            topk=topk,
            earlyStopNum=int(slrcParams["early_stop_num"]),
        )
        trainer.init_hyperparams(
            model=buildSlrcModel(corpus, int(slrcParams["emb_size"])),
            batch_size=int(slrcParams["batch_size"]),
            lr=float(slrcParams["lr"]),
            l2_reg_coef=float(slrcParams["l2_reg_coef"]),
        )
        trainer.train()
        slrcDevMetrics = toFloatMetrics(trainer.evaluate(mode="dev"))
        slrcTestMetrics = toFloatMetrics(trainer.evaluate(mode="test"))
        devUserEmb = trainer.get_predictions(mode="dev").astype("float32", copy=False)
        testUserEmb = trainer.get_predictions(mode="test").astype("float32", copy=False)

    return {
        "slrcDevMetrics": slrcDevMetrics,
        "slrcTestMetrics": slrcTestMetrics,
        "devUserEmb": devUserEmb,
        "testUserEmb": testUserEmb,
    }


def evaluateKnnStage(
    corpus: Corpus,
    userEmb: np.ndarray,
    nearestNeighborsNum: int,
    alpha: float,
    mode: str,
    topk: int,
) -> dict:
    trainer = buildTrainer(
        corpus=corpus,
        maxEpochs=None,
        topk=topk,
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
    return toFloatMetrics(trainer.evaluate(mode=mode))


def runTaiwExperiment(
    dataset: str,
    profileName: str,
    seed: int,
    topk: int,
    slrcParams: dict,
    nbrknnParams: dict,
    workDir: Path,
    profileRootOverride: str | None = None,
) -> dict:
    corpus, resolvedPrefix = buildCorpus(
        dataset=dataset,
        profileName=profileName,
        profileRootOverride=profileRootOverride,
    )

    slrcResult = runSlrcStage(
        corpus=corpus,
        seed=seed,
        topk=topk,
        slrcParams=slrcParams,
        workDir=workDir,
    )
    taiwDevMetrics = evaluateKnnStage(
        corpus=corpus,
        userEmb=slrcResult["devUserEmb"],
        nearestNeighborsNum=int(nbrknnParams["nearest_neighbors_num"]),
        alpha=float(nbrknnParams["alpha"]),
        mode="dev",
        topk=topk,
    )
    taiwTestMetrics = evaluateKnnStage(
        corpus=corpus,
        userEmb=slrcResult["testUserEmb"],
        nearestNeighborsNum=int(nbrknnParams["nearest_neighbors_num"]),
        alpha=float(nbrknnParams["alpha"]),
        mode="test",
        topk=topk,
    )
    return {
        "dataset": dataset,
        "profileName": profileName,
        "profileRoot": str(getProfileRoot(profileName)),
        "resolvedCorpusPrefix": str(resolvedPrefix),
        "seed": int(seed),
        "topk": int(topk),
        "slrcParams": {
            "emb_size": int(slrcParams["emb_size"]),
            "batch_size": int(slrcParams["batch_size"]),
            "lr": float(slrcParams["lr"]),
            "l2_reg_coef": float(slrcParams["l2_reg_coef"]),
            "max_epochs": int(slrcParams["max_epochs"]),
            "early_stop_num": int(slrcParams["early_stop_num"]),
        },
        "nbrknnParams": {
            "nearest_neighbors_num": int(nbrknnParams["nearest_neighbors_num"]),
            "alpha": float(nbrknnParams["alpha"]),
        },
        "slrcDevMetrics": slrcResult["slrcDevMetrics"],
        "slrcTestMetrics": slrcResult["slrcTestMetrics"],
        "taiwDevMetrics": taiwDevMetrics,
        "taiwTestMetrics": taiwTestMetrics,
    }
