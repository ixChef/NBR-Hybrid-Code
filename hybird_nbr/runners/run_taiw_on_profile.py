import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


projectRoot = Path(__file__).resolve().parents[1]
workspaceRoot = projectRoot.parent
taiwRepoRoot = workspaceRoot / "time_aware_item_weighting"

sys.path.insert(0, str(taiwRepoRoot))

from nbr.preparation.corpus import Corpus
from nbr.trainer.trainer import NBRTrainer
from nbr.model.bpr import BPR
from nbr.model.slrc import SLRC
from nbr.model.nbrknn import NBRKNN


def setSeed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def buildSlrcModel(corpus, embSize: int):
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


def buildTrainer(corpus, maxEpochs, topk, earlyStopNum):
    return NBRTrainer(
        corpus=corpus,
        max_epochs=maxEpochs,
        topk=topk,
        early_stop_num=earlyStopNum,
    )


def evaluateKnnStage(corpus, userEmb, nearestNeighborsNum: int, alpha: float, mode: str, topk: int):
    trainer = buildTrainer(
        corpus=corpus,
        maxEpochs=None,
        topk=topk,
        earlyStopNum=None,
    )

    knnModel = NBRKNN(
        item_num=corpus.n_items,
        user_num=corpus.n_users,
        nearest_neighbors_num=nearestNeighborsNum,
        alpha=alpha,
        user_emb=userEmb,
    )

    trainer.init_hyperparams(model=knnModel)
    metrics = trainer.evaluate(mode=mode)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["dunnhumby", "tafeng"])
    parser.add_argument(
        "--profile-root",
        default=str(projectRoot / "data" / "taiw_profile"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(projectRoot / "results"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topk", type=int, default=10)

    parser.add_argument("--emb-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0006142297613045982)
    parser.add_argument("--l2-reg-coef", type=float, default=0.0047331742711911855)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--early-stop-num", type=int, default=3)

    parser.add_argument("--nearest-neighbors-num", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.5)

    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    setSeed(args.seed)

    outputDir = Path(args.output_dir)
    outputDir.mkdir(parents=True, exist_ok=True)

    corpus = Corpus(args.profile_root, args.dataset)
    corpus.load_data()

    slrcTrainer = buildTrainer(
        corpus=corpus,
        maxEpochs=args.max_epochs,
        topk=args.topk,
        earlyStopNum=args.early_stop_num,
    )

    slrcTrainer.init_hyperparams(
        model=buildSlrcModel(corpus, args.emb_size),
        batch_size=args.batch_size,
        lr=args.lr,
        l2_reg_coef=args.l2_reg_coef,
    )

    print("trainer device:", slrcTrainer.device)
    print("model device:", next(slrcTrainer.model.parameters()).device)

    if not args.skip_train:
        slrcTrainer.train()

    slrcDevMetrics = slrcTrainer.evaluate(mode="dev")
    slrcTestMetrics = slrcTrainer.evaluate(mode="test")

    devUserEmb = slrcTrainer.get_predictions(mode="dev")
    testUserEmb = slrcTrainer.get_predictions(mode="test")

    taiwDevMetrics = evaluateKnnStage(
        corpus=corpus,
        userEmb=devUserEmb,
        nearestNeighborsNum=args.nearest_neighbors_num,
        alpha=args.alpha,
        mode="dev",
        topk=args.topk,
    )

    taiwTestMetrics = evaluateKnnStage(
        corpus=corpus,
        userEmb=testUserEmb,
        nearestNeighborsNum=args.nearest_neighbors_num,
        alpha=args.alpha,
        mode="test",
        topk=args.topk,
    )

    results = {
        "dataset": args.dataset,
        "profileRoot": args.profile_root,
        "seed": args.seed,
        "topk": args.topk,
        "slrcParams": {
            "emb_size": args.emb_size,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "l2_reg_coef": args.l2_reg_coef,
            "max_epochs": args.max_epochs,
            "early_stop_num": args.early_stop_num,
        },
        "nbrknnParams": {
            "nearest_neighbors_num": args.nearest_neighbors_num,
            "alpha": args.alpha,
        },
        "slrcDevMetrics": slrcDevMetrics,
        "slrcTestMetrics": slrcTestMetrics,
        "taiwDevMetrics": taiwDevMetrics,
        "taiwTestMetrics": taiwTestMetrics,
    }

    outPath = outputDir / f"taiw_profile_{args.dataset}_taiw_results.json"
    with open(outPath, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSLRC dev:", slrcDevMetrics)
    print("SLRC test:", slrcTestMetrics)
    print("TAIW dev:", taiwDevMetrics)
    print("TAIW test:", taiwTestMetrics)
    print("\nSaved to:", outPath)


if __name__ == "__main__":
    main()