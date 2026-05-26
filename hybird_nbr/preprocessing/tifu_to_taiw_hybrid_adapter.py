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


def normaliseSplitDf(df: pd.DataFrame, splitName: str):
    out = df.copy().reset_index(drop=True)
    out["user_id"] = out["user_id"].astype(int)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="raise")
    out["time_int"] = (out["timestamp"].astype("int64") // 10**9).astype(int)
    out["basket"] = out["basket"].apply(parseBasket)
    out["split_name"] = splitName
    out["source_order"] = out.index.astype(int)
    return out[["user_id", "timestamp", "time_int", "basket", "split_name", "source_order"]]


def validateEvalSplit(df: pd.DataFrame, splitName: str, dataset: str):
    duplicatedUsers = df["user_id"][df["user_id"].duplicated()].unique().tolist()
    if duplicatedUsers:
        preview = duplicatedUsers[:10]
        raise ValueError(
            f"{dataset}: {splitName} split is expected to contain at most one basket per user for hybrid alignment. "
            f"Found duplicated users such as: {preview}"
        )


def buildHybridTaiwBasketsFromTifuSplits(trainDf: pd.DataFrame, valDf: pd.DataFrame, testDf: pd.DataFrame, dataset: str):
    trainBaskets = normaliseSplitDf(trainDf, "train")
    devBaskets = normaliseSplitDf(valDf, "dev")
    testBaskets = normaliseSplitDf(testDf, "test")

    validateEvalSplit(devBaskets, "validate", dataset)
    validateEvalSplit(testBaskets, "test", dataset)

    baskets = pd.concat(
        [
            trainBaskets,
            devBaskets,
            testBaskets,
        ],
        ignore_index=True,
    )

    splitOrder = {"train": 0, "dev": 1, "test": 2}
    baskets["split_order"] = baskets["split_name"].map(splitOrder)

    baskets = baskets.sort_values(
        ["user_id", "time_int", "split_order", "source_order"],
        kind="stable",
    ).reset_index(drop=True)

    baskets["basket_order"] = baskets.groupby("user_id").cumcount().astype(int)
    return baskets, devBaskets, testBaskets


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
                    "source_order": int(row.source_order),
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


def buildEvalDf(clicks: pd.DataFrame, splitName: str, orderedUsers: pd.Series):
    grouped = (
        clicks.loc[clicks["split_name"] == splitName, ["user_id", "seq_order"]]
        .groupby("user_id", sort=False)["seq_order"]
        .apply(lambda xs: [int(x) for x in xs.tolist()])
        .reset_index(name="gt_order")
    )

    orderDf = pd.DataFrame(
        {
            "user_id": orderedUsers.astype(int).tolist(),
            "_eval_order": list(range(len(orderedUsers))),
        }
    )

    evalDf = orderDf.merge(grouped, on="user_id", how="left")
    missingMask = evalDf["gt_order"].isna()
    if missingMask.any():
        missingUsers = evalDf.loc[missingMask, "user_id"].head(10).tolist()
        raise ValueError(f"Missing gt_order rows for users in {splitName}: {missingUsers}")

    evalDf = evalDf.sort_values("_eval_order", kind="stable").reset_index(drop=True)
    evalDf = evalDf[["user_id", "gt_order"]]
    return evalDf


def validateHybridShape(trainDf: pd.DataFrame, devDf: pd.DataFrame, testDf: pd.DataFrame, baskets: pd.DataFrame, dataset: str):
    basketCounts = baskets["split_name"].value_counts().to_dict()

    expectedTrain = int(trainDf.shape[0])
    expectedDev = int(devDf.shape[0])
    expectedTest = int(testDf.shape[0])

    actualTrain = int(basketCounts.get("train", 0))
    actualDev = int(basketCounts.get("dev", 0))
    actualTest = int(basketCounts.get("test", 0))

    if actualTrain != expectedTrain:
        raise ValueError(f"{dataset}: expected {expectedTrain} train baskets, found {actualTrain}")
    if actualDev != expectedDev:
        raise ValueError(f"{dataset}: expected {expectedDev} dev baskets, found {actualDev}")
    if actualTest != expectedTest:
        raise ValueError(f"{dataset}: expected {expectedTest} test baskets, found {actualTest}")


def writeTaiwFiles(outputRoot: Path, dataset: str, bookDf: pd.DataFrame, trainDf: pd.DataFrame, devDf: pd.DataFrame, testDf: pd.DataFrame):
    datasetDir = outputRoot / f"data_{dataset}"
    datasetDir.mkdir(parents=True, exist_ok=True)

    bookDf.to_csv(datasetDir / "book.csv", sep="\t", index=False)
    trainDf.to_csv(datasetDir / "train.csv", sep="\t", index=False)
    devDf.to_csv(datasetDir / "dev.csv", sep="\t", index=False)
    testDf.to_csv(datasetDir / "test.csv", sep="\t", index=False)

    return datasetDir


def convertTifuProfileToTaiwHybrid(tifuRoot: Path, outputRoot: Path, dataset: str):
    trainSplit, valSplit, testSplit = loadSplitFiles(tifuRoot, dataset)

    baskets, devBaskets, testBaskets = buildHybridTaiwBasketsFromTifuSplits(
        trainSplit,
        valSplit,
        testSplit,
        dataset,
    )
    validateHybridShape(trainSplit, devBaskets, testBaskets, baskets, dataset)

    clicks = explodeClicks(baskets)

    bookDf = buildBookDf(clicks)
    trainDf = buildTrainDf(clicks)
    devDf = buildEvalDf(clicks, "dev", devBaskets["user_id"])
    testDf = buildEvalDf(clicks, "test", testBaskets["user_id"])

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
    )
    parser.add_argument(
        "--output-root",
        default="/notebooks/A2FYP/hybrid_nbr/data/tifu_profile_taiw_hybrid",
    )
    args = parser.parse_args()

    summary = convertTifuProfileToTaiwHybrid(
        tifuRoot=Path(args.tifu_root),
        outputRoot=Path(args.output_root),
        dataset=args.dataset,
    )

    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()