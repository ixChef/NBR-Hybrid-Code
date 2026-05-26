import argparse
import json
from pathlib import Path

import optuna
from optuna.trial import TrialState

projectRoot = Path(__file__).resolve().parents[1]
tifuRepoRoot = projectRoot.parent / "time_dependent_nbr"

def getStudyInfo(studyName: str):
    dbPath = tifuRepoRoot / "results" / f"{studyName}.db"
    info = {
        "studyName": studyName,
        "dbPath": str(dbPath),
        "exists": dbPath.exists(),
    }

    if not dbPath.exists():
        return info

    study = optuna.load_study(
        study_name=studyName,
        storage=f"sqlite:///{dbPath}",
    )

    completeTrials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    failedTrials = [t for t in study.trials if t.state == TrialState.FAIL]
    prunedTrials = [t for t in study.trials if t.state == TrialState.PRUNED]
    runningTrials = [t for t in study.trials if t.state == TrialState.RUNNING]
    waitingTrials = [t for t in study.trials if t.state == TrialState.WAITING]
    finishedTrials = completeTrials + failedTrials + prunedTrials

    info.update(
        {
            "totalTrials": len(study.trials),
            "finishedTrials": len(finishedTrials),
            "completeTrials": len(completeTrials),
            "failedTrials": len(failedTrials),
            "prunedTrials": len(prunedTrials),
            "runningTrials": len(runningTrials),
            "waitingTrials": len(waitingTrials),
        }
    )

    if study.trials:
        lastTrial = study.trials[-1]
        info["lastTrialNumber"] = lastTrial.number
        info["lastTrialState"] = lastTrial.state.name
        info["lastTrialValue"] = lastTrial.value

    if completeTrials:
        info["bestTrialNumber"] = study.best_trial.number
        info["bestValue"] = study.best_value
        info["bestParams"] = study.best_params

    return info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", required=True)
    args = parser.parse_args()

    info = getStudyInfo(args.study_name)
    print(json.dumps(info, indent=2, sort_keys=False))

if __name__ == "__main__":
    main()