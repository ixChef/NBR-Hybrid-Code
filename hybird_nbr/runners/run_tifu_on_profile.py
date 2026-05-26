import argparse
import sys
from pathlib import Path

projectRoot = Path(__file__).resolve().parents[1]
workspaceRoot = projectRoot.parent
tifuRepoRoot = workspaceRoot / "time_dependent_nbr"

sys.path.insert(0, str(tifuRepoRoot))

import src.settings as tifuSettings
import src.dataset.base as tifuBaseModule
import src.scripts.experiment as tifuExperiment
from src.models import MODELS


def ensureAlias(profileRoot: Path, datasetName: str, aliasName: str):
    sourceDir = profileRoot / datasetName
    aliasDir = profileRoot / aliasName

    if not sourceDir.exists():
        raise FileNotFoundError(f"Missing dataset directory: {sourceDir}")

    if aliasDir.exists() or aliasDir.is_symlink():
        return aliasName

    aliasDir.symlink_to(sourceDir.resolve(), target_is_directory=True)
    return aliasName


def resolveModelKey(modelName: str):
    if modelName in MODELS:
        return modelName

    classAliases = {
        "tifuknn_td": "TIFUKNNTimeDaysNextTsRecommender",
        "tifu_td": "TIFUKNNTimeDaysNextTsRecommender",
        "tifuknn_time_days_next_ts": "TIFUKNNTimeDaysNextTsRecommender",
    }

    if modelName in classAliases:
        targetClassName = classAliases[modelName]
        matches = [key for key, cls in MODELS.items() if cls.__name__ == targetClassName]

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            raise ValueError(
                f"Alias {modelName} matched multiple model keys for {targetClassName}: {sorted(matches)}"
            )

    availableKeys = sorted(MODELS.keys())
    raise ValueError(
        f"Unknown model alias/key: {modelName}. Available model keys: {availableKeys}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["dunnhumby", "tafeng"])
    parser.add_argument("--model", default="tifuknn_td")
    parser.add_argument("--metric", default="recall")
    parser.add_argument("--cutoff", type=int, default=10)
    parser.add_argument("--num-trials", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--profile-root",
        default=str(projectRoot / "data" / "taiw_profile_tifu"),
    )
    parser.add_argument("--alias-prefix", default="taiw_profile")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    profileRoot = Path(args.profile_root)
    aliasName = f"{args.alias_prefix}_{args.dataset}"
    ensureAlias(profileRoot, args.dataset, aliasName)

    resolvedModel = resolveModelKey(args.model)

    tifuSettings.DATA_DIR = str(profileRoot)
    tifuBaseModule.DATA_DIR = str(profileRoot)
    tifuExperiment.DATA_DIR = str(profileRoot)

    print(f"profileRoot: {profileRoot}")
    print(f"dataset: {args.dataset}")
    print(f"datasetAlias: {aliasName}")
    print(f"modelArg: {args.model}")
    print(f"resolvedModel: {resolvedModel}")
    print(f"numTrials: {args.num_trials}")

    tifuExperiment.run_experiment(
        dataset=args.dataset,
        model=resolvedModel,
        metric=args.metric,
        cutoff=args.cutoff,
        num_trials=args.num_trials,
        batch_size=args.batch_size,
        dataset_dir_name=aliasName,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()