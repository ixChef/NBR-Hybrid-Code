from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import optuna
from optuna.trial import TrialState

projectRoot = Path(__file__).resolve().parents[1]
if str(projectRoot) not in sys.path:
    sys.path.insert(0, str(projectRoot))

from configs.taiw_route_a import defaultEvalSeeds, paperSlrcParams
from runners.taiw_common import runTaiwExperiment, resolveCorpusPrefix


def writeJson(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def appendJsonLine(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def getRunDir(dataset: str, profileName: str, searchSpace: str) -> Path:
    return projectRoot / "results" / "taiw_route_a" / searchSpace / profileName / dataset


def getStorageUrl(runDir: Path) -> str:
    return f"sqlite:///{(runDir / 'optuna.db').as_posix()}"


def getStudyName(dataset: str, profileName: str, searchSpace: str, tuneSeed: int) -> str:
    return f"taiw_route_a__{searchSpace}__{profileName}__{dataset}__seed{tuneSeed}"


def getStateCounts(study: optuna.Study) -> dict:
    counts = Counter(trial.state.name for trial in study.trials)
    return {state: int(counts.get(state, 0)) for state in ["COMPLETE", "RUNNING", "FAIL", "PRUNED", "WAITING"]}


def countCompleteTrials(study: optuna.Study) -> int:
    return sum(1 for trial in study.trials if trial.state == TrialState.COMPLETE)


def getBestBundle(dataset: str, searchSpace: str, params: dict) -> tuple[dict, dict]:
    if searchSpace == "knn_only":
        if dataset not in paperSlrcParams:
            raise ValueError(f"No paper SLRC params configured for dataset={dataset}")
        slrcParams = dict(paperSlrcParams[dataset])
    else:
        slrcParams = {
            "emb_size": int(params["emb_size"]),
            "batch_size": int(params["batch_size"]),
            "lr": float(params["lr"]),
            "l2_reg_coef": float(params["l2_reg_coef"]),
            "max_epochs": 20,
            "early_stop_num": 3,
        }

    nbrknnParams = {
        "nearest_neighbors_num": int(params["nearest_neighbors_num"]),
        "alpha": float(params["alpha"]),
    }
    return slrcParams, nbrknnParams


def suggestParams(trial: optuna.Trial, dataset: str, searchSpace: str) -> tuple[dict, dict]:
    if searchSpace == "knn_only":
        if dataset not in paperSlrcParams:
            raise ValueError(f"No paper SLRC params configured for dataset={dataset}")
        slrcParams = dict(paperSlrcParams[dataset])
    else:
        slrcParams = {
            "emb_size": int(trial.suggest_categorical("emb_size", [32, 64])),
            "batch_size": int(trial.suggest_categorical("batch_size", [64, 128, 256])),
            "lr": float(trial.suggest_float("lr", 1e-5, 1e-2, log=True)),
            "l2_reg_coef": float(trial.suggest_float("l2_reg_coef", 1e-5, 1e-1, log=True)),
            "max_epochs": 20,
            "early_stop_num": 3,
        }

    nbrknnParams = {
        "nearest_neighbors_num": int(trial.suggest_int("nearest_neighbors_num", 1, 200)),
        "alpha": float(trial.suggest_float("alpha", 0.0, 1.0, step=0.05)),
    }
    return slrcParams, nbrknnParams


def buildProgressPayload(
    study: optuna.Study,
    dataset: str,
    profileName: str,
    searchSpace: str,
    tuneSeed: int,
    targetTrials: int,
) -> dict:
    stateCounts = getStateCounts(study)
    payload = {
        "dataset": dataset,
        "profileName": profileName,
        "searchSpace": searchSpace,
        "tuneSeed": int(tuneSeed),
        "targetCompleteTrials": int(targetTrials),
        "completeTrials": int(stateCounts["COMPLETE"]),
        "stateCounts": stateCounts,
        "isFinished": bool(stateCounts["COMPLETE"] >= targetTrials),
        "studyName": study.study_name,
    }
    if stateCounts["COMPLETE"] > 0:
        payload["bestValue"] = float(study.best_value)
        payload["bestParams"] = dict(study.best_params)
        payload["bestTrialNumber"] = int(study.best_trial.number)
    return payload


def persistStudyState(
    runDir: Path,
    study: optuna.Study,
    dataset: str,
    profileName: str,
    searchSpace: str,
    tuneSeed: int,
    targetTrials: int,
    latestTrial: optuna.trial.FrozenTrial | None = None,
) -> None:
    progressPayload = buildProgressPayload(
        study=study,
        dataset=dataset,
        profileName=profileName,
        searchSpace=searchSpace,
        tuneSeed=tuneSeed,
        targetTrials=targetTrials,
    )
    writeJson(runDir / "progress.json", progressPayload)

    if countCompleteTrials(study) > 0:
        slrcParams, nbrknnParams = getBestBundle(dataset, searchSpace, study.best_params)
        bestPayload = {
            "dataset": dataset,
            "profileName": profileName,
            "searchSpace": searchSpace,
            "tuneSeed": int(tuneSeed),
            "studyName": study.study_name,
            "bestTrialNumber": int(study.best_trial.number),
            "bestValue": float(study.best_value),
            "bestParams": dict(study.best_params),
            "slrcParams": slrcParams,
            "nbrknnParams": nbrknnParams,
        }
        writeJson(runDir / "best_trial.json", bestPayload)

    if latestTrial is not None:
        line = {
            "trialNumber": int(latestTrial.number),
            "state": latestTrial.state.name,
            "value": None if latestTrial.value is None else float(latestTrial.value),
            "params": dict(latestTrial.params),
        }
        appendJsonLine(runDir / "tune_log.jsonl", line)


def makeObjective(
    dataset: str,
    profileName: str,
    searchSpace: str,
    topk: int,
    tuneSeed: int,
    runDir: Path,
    profileRootOverride: str | None,
):
    def objective(trial: optuna.Trial) -> float:
        slrcParams, nbrknnParams = suggestParams(trial, dataset=dataset, searchSpace=searchSpace)
        trialWorkDir = runDir / "trial_work" / f"trial_{trial.number:04d}"
        startedAt = time.time()
        result = runTaiwExperiment(
            dataset=dataset,
            profileName=profileName,
            seed=tuneSeed,
            topk=topk,
            slrcParams=slrcParams,
            nbrknnParams=nbrknnParams,
            workDir=trialWorkDir,
            profileRootOverride=profileRootOverride,
        )
        score = float(result["taiwDevMetrics"]["ndcg"])
        trial.set_user_attr("resolvedCorpusPrefix", result["resolvedCorpusPrefix"])
        trial.set_user_attr("slrcParams", result["slrcParams"])
        trial.set_user_attr("nbrknnParams", result["nbrknnParams"])
        trial.set_user_attr("slrcDevMetrics", result["slrcDevMetrics"])
        trial.set_user_attr("slrcTestMetrics", result["slrcTestMetrics"])
        trial.set_user_attr("taiwDevMetrics", result["taiwDevMetrics"])
        trial.set_user_attr("taiwTestMetrics", result["taiwTestMetrics"])
        trial.set_user_attr("elapsedMin", float((time.time() - startedAt) / 60.0))
        writeJson(
            runDir / "trials" / f"trial_{trial.number:04d}.json",
            {
                "trialNumber": int(trial.number),
                "value": score,
                "params": dict(trial.params),
                "result": result,
            },
        )
        return score

    return objective


def loadStudy(runDir: Path, dataset: str, profileName: str, searchSpace: str, tuneSeed: int) -> optuna.Study:
    storageUrl = getStorageUrl(runDir)
    studyName = getStudyName(dataset, profileName, searchSpace, tuneSeed)
    sampler = optuna.samplers.TPESampler(seed=tuneSeed)
    return optuna.create_study(
        direction="maximize",
        sampler=sampler,
        storage=storageUrl,
        study_name=studyName,
        load_if_exists=True,
    )


def preflightCorpus(args: argparse.Namespace) -> Path:
    resolvedPrefix = resolveCorpusPrefix(
        profileName=args.profile,
        dataset=args.dataset,
        profileRootOverride=args.profile_root_override,
    )
    payload = {
        "dataset": args.dataset,
        "profileName": args.profile,
        "resolvedCorpusPrefix": str(resolvedPrefix),
        "expectedBookPath": str(resolvedPrefix / f"data_{args.dataset}" / "book.csv"),
    }
    print(json.dumps(payload, indent=2))
    return resolvedPrefix


def runTune(args: argparse.Namespace) -> None:
    preflightCorpus(args)

    runDir = getRunDir(args.dataset, args.profile, args.search_space)
    runDir.mkdir(parents=True, exist_ok=True)

    study = loadStudy(
        runDir=runDir,
        dataset=args.dataset,
        profileName=args.profile,
        searchSpace=args.search_space,
        tuneSeed=args.tune_seed,
    )

    objective = makeObjective(
        dataset=args.dataset,
        profileName=args.profile,
        searchSpace=args.search_space,
        topk=args.topk,
        tuneSeed=args.tune_seed,
        runDir=runDir,
        profileRootOverride=args.profile_root_override,
    )

    while countCompleteTrials(study) < args.target_trials:
        study.optimize(
            objective,
            n_trials=1,
            gc_after_trial=True,
            catch=(Exception,),
        )
        latestTrial = study.trials[-1]
        persistStudyState(
            runDir=runDir,
            study=study,
            dataset=args.dataset,
            profileName=args.profile,
            searchSpace=args.search_space,
            tuneSeed=args.tune_seed,
            targetTrials=args.target_trials,
            latestTrial=latestTrial,
        )

    persistStudyState(
        runDir=runDir,
        study=study,
        dataset=args.dataset,
        profileName=args.profile,
        searchSpace=args.search_space,
        tuneSeed=args.tune_seed,
        targetTrials=args.target_trials,
        latestTrial=None,
    )
    print(json.dumps(buildProgressPayload(
        study=study,
        dataset=args.dataset,
        profileName=args.profile,
        searchSpace=args.search_space,
        tuneSeed=args.tune_seed,
        targetTrials=args.target_trials,
    ), indent=2))


def runStatus(args: argparse.Namespace) -> None:
    runDir = getRunDir(args.dataset, args.profile, args.search_space)
    if not (runDir / "optuna.db").exists():
        print(json.dumps({"message": "No study found yet.", "runDir": str(runDir)}, indent=2))
        return

    study = loadStudy(
        runDir=runDir,
        dataset=args.dataset,
        profileName=args.profile,
        searchSpace=args.search_space,
        tuneSeed=args.tune_seed,
    )
    progressPayload = buildProgressPayload(
        study=study,
        dataset=args.dataset,
        profileName=args.profile,
        searchSpace=args.search_space,
        tuneSeed=args.tune_seed,
        targetTrials=args.target_trials,
    )
    writeJson(runDir / "progress.json", progressPayload)
    print(json.dumps(progressPayload, indent=2))


def metricSummary(values: list[float]) -> dict:
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)),
    }


def runEval(args: argparse.Namespace) -> None:
    preflightCorpus(args)

    runDir = getRunDir(args.dataset, args.profile, args.search_space)
    if not (runDir / "optuna.db").exists():
        raise RuntimeError(f"No study found at {runDir / 'optuna.db'}")

    study = loadStudy(
        runDir=runDir,
        dataset=args.dataset,
        profileName=args.profile,
        searchSpace=args.search_space,
        tuneSeed=args.tune_seed,
    )
    if countCompleteTrials(study) == 0:
        raise RuntimeError("Study exists but has no COMPLETE trials.")

    slrcParams, nbrknnParams = getBestBundle(args.dataset, args.search_space, study.best_params)

    finalEvalDir = runDir / "final_eval"
    finalEvalDir.mkdir(parents=True, exist_ok=True)

    precisionValues = []
    recallValues = []
    ndcgValues = []

    for evalSeed in args.seeds:
        seedPath = finalEvalDir / f"seed_{int(evalSeed)}.json"
        if seedPath.exists() and not args.force:
            payload = json.loads(seedPath.read_text())
        else:
            seedWorkDir = runDir / "eval_work" / f"seed_{int(evalSeed)}"
            result = runTaiwExperiment(
                dataset=args.dataset,
                profileName=args.profile,
                seed=int(evalSeed),
                topk=args.topk,
                slrcParams=slrcParams,
                nbrknnParams=nbrknnParams,
                workDir=seedWorkDir,
                profileRootOverride=args.profile_root_override,
            )
            payload = {
                "evalSeed": int(evalSeed),
                "studyName": study.study_name,
                "bestTrialNumber": int(study.best_trial.number),
                "bestValue": float(study.best_value),
                "bestParams": dict(study.best_params),
                "slrcParams": slrcParams,
                "nbrknnParams": nbrknnParams,
                "slrcDevMetrics": result["slrcDevMetrics"],
                "slrcTestMetrics": result["slrcTestMetrics"],
                "taiwDevMetrics": result["taiwDevMetrics"],
                "taiwTestMetrics": result["taiwTestMetrics"],
                "resolvedCorpusPrefix": result["resolvedCorpusPrefix"],
            }
            writeJson(seedPath, payload)

        testMetrics = payload["taiwTestMetrics"]
        precisionValues.append(float(testMetrics["precision"]))
        recallValues.append(float(testMetrics["recall"]))
        ndcgValues.append(float(testMetrics["ndcg"]))

    averaged = {
        "dataset": args.dataset,
        "profileName": args.profile,
        "searchSpace": args.search_space,
        "topk": int(args.topk),
        "tuneSeed": int(args.tune_seed),
        "evalSeeds": [int(seed) for seed in args.seeds],
        "nSeeds": len(args.seeds),
        "studyName": study.study_name,
        "bestTrialNumber": int(study.best_trial.number),
        "bestValue": float(study.best_value),
        "bestParams": dict(study.best_params),
        "slrcParams": slrcParams,
        "nbrknnParams": nbrknnParams,
        "precision": metricSummary(precisionValues),
        "recall": metricSummary(recallValues),
        "ndcg": metricSummary(ndcgValues),
    }
    writeJson(finalEvalDir / "averaged_metrics.json", averaged)
    print(json.dumps(averaged, indent=2))


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    def addCommonFlags(target: argparse.ArgumentParser) -> None:
        target.add_argument("--dataset", required=True, choices=["dunnhumby", "tafeng"])
        target.add_argument("--profile", required=True, choices=["tifu_profile", "taiw_profile", "tifu_hybrid_profile"])
        target.add_argument("--profile-root-override", default=None)
        target.add_argument("--search-space", default="full", choices=["full", "knn_only"])
        target.add_argument("--topk", type=int, default=10)
        target.add_argument("--tune-seed", type=int, default=10)
        target.add_argument("--target-trials", type=int, default=25)

    tuneParser = subparsers.add_parser("tune")
    addCommonFlags(tuneParser)
    tuneParser.set_defaults(func=runTune)

    statusParser = subparsers.add_parser("status")
    addCommonFlags(statusParser)
    statusParser.set_defaults(func=runStatus)

    evalParser = subparsers.add_parser("eval")
    addCommonFlags(evalParser)
    evalParser.add_argument("--seeds", type=int, nargs="+", default=defaultEvalSeeds)
    evalParser.add_argument("--force", action="store_true")
    evalParser.set_defaults(func=runEval)

    return parser


def main() -> None:
    parser = buildParser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
