import argparse
import ast
from pathlib import Path

import pandas as pd


def parseBasket(value):
    if isinstance(value, list):
        return [int(x) for x in value]
    if pd.isna(value):
        return []
    return [int(x) for x in ast.literal_eval(value)]


def loadSplitFiles(tifuRoot: Path, dataset: str):
    datasetRoot = tifuRoot / dataset
    trainPath = datasetRoot / "split" / "train.csv"
    valPath = datasetRoot / "split" / "validate.csv"
    testPath = datasetRoot / "split" / "test.csv"

    for path in [trainPath, valPath, testPath]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required split file: {path}")

    trainDf = pd.read_csv(trainPath)
    valDf = pd.read_csv(valPath)
    testDf = pd.read_csv(testPath)

    return trainDf, valDf, testDf


def normalizeSplitDf(df: pd.DataFrame, splitName: str):
    out = df.copy()
    out["user_id"] = out["user_id"].astype(int)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="raise")
    out["time_int"] = (out["timestamp"].astype("int64") // 10**9).astype(int)
    out["basket"] = out["basket"].apply(parseBasket)
    out["split_name"] = splitName
    return out[["user_id", "timestamp", "time_int", "basket", "split_name"]]


def buildTaiwBasketsFromTifuSplits(trainDf: pd.DataFrame, valDf: pd.DataFrame, testDf: pd.DataFrame, dataset: str):
    trainBaskets = normalizeSplitDf(trainDf, "train")
    heldoutBaskets = pd.concat(
        [
            normalizeSplitDf(valDf, "heldout"),
            normalizeSplitDf(testDf, "heldout"),
        ],
        ignore_index=True,
    )

    trainBaskets = trainBaskets.sort_values(
        ["user_id", "time_int"],
        kind="stable",
    ).reset_index(drop=True)

    heldoutBaskets = heldoutBaskets.sort_values(
        ["user_id", "time_int"],
        kind="stable",
    ).reset_index(drop=True)

    heldoutCounts = heldoutBaskets.groupby("user_id").size()
    badHeldout = heldoutCounts[heldoutCounts != 1]
    if len(badHeldout) > 0:
        raise ValueError(
            f"{dataset}: expected exactly one heldout basket per user from TIFU split.\n"
            f"{badHeldout.head(10).to_string()}"
        )

    trainCounts = trainBaskets.groupby("user_id").size()
    badTrain = trainCounts[trainCounts < 1]
    if len(badTrain) > 0:
        raise ValueError(
            f"{dataset}: some users have no train basket, cannot derive TAIW dev basket.\n"
            f"{badTrain.head(10).to_string()}"
        )

    trainBaskets["train_index"] = trainBaskets.groupby("user_id").cumcount().astype(int)
    trainBaskets["train_count"] = trainBaskets.groupby("user_id")["user_id"].transform("size").astype(int)

    devMask = trainBaskets["train_index"] == (trainBaskets["train_count"] - 1)
    trainBaskets.loc[devMask, "split_name"] = "dev"

    heldoutBaskets["split_name"] = "test"

    baskets = pd.concat(
        [
            trainBaskets[["user_id", "timestamp", "time_int", "basket", "split_name"]],
            heldoutBaskets[["user_id", "timestamp", "time_int", "basket", "split_name"]],
        ],
        ignore_index=True,
    )

    splitOrder = {"train": 0, "dev": 1, "test": 2}
    baskets["split_order"] = baskets["split_name"].map(splitOrder)

    baskets = baskets.sort_values(
        ["user_id", "time_int", "split_order"],
        kind="stable",
    ).reset_index(drop=True)

    baskets["basket_order"] = baskets.groupby("user_id").cumcount().astype(int)
    return baskets


def explodeClicks(baskets: pd.DataFrame):
    records = []

    for row in baskets.itertuples(index=False):
        for itemPos, itemId in enumerate(row.basket):
            records.append(
                {
                    "user_id": int(row.user_id),
                    "time_int": int(row.time_int),
                    "split_name": row.split_name,
                    "basket_order": int(row.basket_order),
                    "item_pos": int(itemPos),
                    "item_id": int(itemId),
                }
            )

    clicks = pd.DataFrame(records)
    clicks = clicks.sort_values(
        ["user_id", "time_int", "basket_order", "item_pos"],
        kind="stable",
    ).reset_index(drop=True)

    clicks["seq_order"] = clicks.groupby("user_id").cumcount().astype(int)
    return clicks


def buildBookDf(clicks: pd.DataFrame):
    bookDf = (
        clicks.groupby("user_id", sort=False)
        .apply(lambda g: [(int(row.item_id), int(row.time_int)) for row in g.itertuples(index=False)])
        .reset_index(name="sequence (item_id,time)")
    )
    return bookDf


def buildTrainDf(clicks: pd.DataFrame):
    trainDf = (
        clicks.loc[clicks["split_name"] == "train", ["user_id", "seq_order"]]
        .rename(columns={"seq_order": "consumption_order"})
        .sort_values(["user_id", "consumption_order"], kind="stable")
        .reset_index(drop=True)
    )
    return trainDf


def buildEvalDf(clicks: pd.DataFrame, splitName: str):
    evalDf = (
        clicks.loc[clicks["split_name"] == splitName, ["user_id", "seq_order"]]
        .groupby("user_id", sort=False)["seq_order"]
        .apply(lambda xs: [int(x) for x in xs.tolist()])
        .reset_index(name="gt_order")
    )
    return evalDf


def validateUserSplitShape(baskets: pd.DataFrame, dataset: str):
    counts = baskets.groupby(["user_id", "split_name"]).size().unstack(fill_value=0)

    missingDev = counts.get("dev", pd.Series(dtype=int))
    missingTest = counts.get("test", pd.Series(dtype=int))

    if (missingDev == 0).any():
        badUsers = counts[missingDev == 0].head(10)
        raise ValueError(
            f"{dataset}: some users have no validation basket after conversion.\n"
            f"{badUsers.to_string()}"
        )

    if (missingTest == 0).any():
        badUsers = counts[missingTest == 0].head(10)
        raise ValueError(
            f"{dataset}: some users have no test basket after conversion.\n"
            f"{badUsers.to_string()}"
        )


def writeTaiwFiles(outputRoot: Path, dataset: str, bookDf: pd.DataFrame, trainDf: pd.DataFrame, devDf: pd.DataFrame, testDf: pd.DataFrame):
    datasetDir = outputRoot / f"data_{dataset}"
    datasetDir.mkdir(parents=True, exist_ok=True)

    bookDf.to_csv(datasetDir / "book.csv", sep="\t", index=False)
    trainDf.to_csv(datasetDir / "train.csv", sep="\t", index=False)
    devDf.to_csv(datasetDir / "dev.csv", sep="\t", index=False)
    testDf.to_csv(datasetDir / "test.csv", sep="\t", index=False)

    return datasetDir


def convertTifuProfileToTaiw(tifuRoot: Path, outputRoot: Path, dataset: str):
    trainSplit, valSplit, testSplit = loadSplitFiles(tifuRoot, dataset)

    baskets = buildTaiwBasketsFromTifuSplits(trainSplit, valSplit, testSplit, dataset)
    validateUserSplitShape(baskets, dataset)

    clicks = explodeClicks(baskets)

    bookDf = buildBookDf(clicks)
    trainDf = buildTrainDf(clicks)
    devDf = buildEvalDf(clicks, "dev")
    testDf = buildEvalDf(clicks, "test")

    datasetDir = writeTaiwFiles(outputRoot, dataset, bookDf, trainDf, devDf, testDf)

    summary = {
        "dataset": dataset,
        "book_users": len(bookDf),
        "train_rows": len(trainDf),
        "dev_rows": len(devDf),
        "test_rows": len(testDf),
        "output_dir": str(datasetDir),
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["dunnhumby", "tafeng"])
    parser.add_argument(
        "--tifu-root",
        default="/notebooks/A2FYP/hybrid_nbr/data/tifu_profile",
        help="Root of tifu_profile outputs"
    )
    parser.add_argument(
        "--output-root",
        default="/notebooks/A2FYP/hybrid_nbr/data/tifu_profile_taiw",
        help="Root where TAIW-style files will be written"
    )
    args = parser.parse_args()

    summary = convertTifuProfileToTaiw(
        tifuRoot=Path(args.tifu_root),
        outputRoot=Path(args.output_root),
        dataset=args.dataset
    )

    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()