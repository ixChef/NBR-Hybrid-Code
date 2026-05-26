import argparse
import ast
import shutil
from pathlib import Path

import pandas as pd


def parseSequence(value):
    if isinstance(value, list):
        sequence = value
    else:
        sequence = ast.literal_eval(value)

    parsed = []
    for pair in sequence:
        if len(pair) != 2:
            raise ValueError(f"Invalid sequence entry: {pair}")
        itemId, timeInt = pair
        parsed.append((int(itemId), int(timeInt)))
    return parsed


def parseIntList(value):
    if isinstance(value, list):
        return [int(x) for x in value]
    if pd.isna(value):
        return []
    return [int(x) for x in ast.literal_eval(value)]


def linkOrCopy(sourcePath: Path, targetPath: Path, useSymlink: bool):
    if targetPath.exists() or targetPath.is_symlink():
        return
    if useSymlink:
        try:
            targetPath.symlink_to(sourcePath.resolve())
            return
        except OSError:
            pass
    if sourcePath.is_dir():
        shutil.copytree(sourcePath, targetPath)
    else:
        shutil.copy2(sourcePath, targetPath)


def loadTaiwFiles(taiwRoot: Path, dataset: str):
    datasetDir = taiwRoot / f"data_{dataset}"

    bookPath = datasetDir / "book.csv"
    trainPath = datasetDir / "train.csv"
    devPath = datasetDir / "dev.csv"
    testPath = datasetDir / "test.csv"

    for path in [bookPath, trainPath, devPath, testPath]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required TAIW file: {path}")

    bookDf = pd.read_csv(bookPath, sep="\t")
    trainDf = pd.read_csv(trainPath, sep="\t")
    devDf = pd.read_csv(devPath, sep="\t")
    testDf = pd.read_csv(testPath, sep="\t")

    return bookDf, trainDf, devDf, testDf


def buildClicksFromBook(bookDf: pd.DataFrame):
    records = []

    for row in bookDf.itertuples(index=False):
        userId = int(getattr(row, "user_id"))
        sequence = parseSequence(getattr(row, "_1"))

        for seqOrder, (itemId, timeInt) in enumerate(sequence):
            records.append(
                {
                    "user_id": userId,
                    "seq_order": int(seqOrder),
                    "item_id": int(itemId),
                    "time_int": int(timeInt),
                }
            )

    clicks = pd.DataFrame(records)
    clicks = clicks.sort_values(["user_id", "seq_order"], kind="stable").reset_index(drop=True)
    return clicks


def explodeEvalOrders(df: pd.DataFrame, splitName: str):
    records = []

    for row in df.itertuples(index=False):
        userId = int(getattr(row, "user_id"))
        gtOrders = parseIntList(getattr(row, "gt_order"))
        for seqOrder in gtOrders:
            records.append(
                {
                    "user_id": userId,
                    "seq_order": int(seqOrder),
                    "split_name": splitName,
                }
            )

    out = pd.DataFrame(records)
    if len(out) == 0:
        return pd.DataFrame(columns=["user_id", "seq_order", "split_name"])
    return out.sort_values(["user_id", "seq_order"], kind="stable").reset_index(drop=True)


def buildSplitAssignments(trainDf: pd.DataFrame, devDf: pd.DataFrame, testDf: pd.DataFrame):
    trainAssignments = (
        trainDf.loc[:, ["user_id", "consumption_order"]]
        .rename(columns={"consumption_order": "seq_order"})
        .assign(split_name="train")
    )

    devAssignments = explodeEvalOrders(devDf, "validate")
    testAssignments = explodeEvalOrders(testDf, "test")

    assignments = pd.concat(
        [trainAssignments, devAssignments, testAssignments],
        ignore_index=True,
    )

    assignments["user_id"] = assignments["user_id"].astype(int)
    assignments["seq_order"] = assignments["seq_order"].astype(int)

    duplicateMask = assignments.duplicated(subset=["user_id", "seq_order"], keep=False)
    if duplicateMask.any():
        badRows = assignments.loc[duplicateMask].sort_values(["user_id", "seq_order"], kind="stable").head(20)
        raise ValueError(
            "Duplicate split assignment found for (user_id, seq_order):\n"
            f"{badRows.to_string(index=False)}"
        )

    assignments = assignments.sort_values(["user_id", "seq_order"], kind="stable").reset_index(drop=True)
    return assignments


def attachSplits(clicks: pd.DataFrame, assignments: pd.DataFrame):
    merged = clicks.merge(assignments, on=["user_id", "seq_order"], how="left")

    if merged["split_name"].isna().any():
        badRows = merged.loc[merged["split_name"].isna(), ["user_id", "seq_order", "item_id", "time_int"]].head(20)
        raise ValueError(
            "Some clicks were not assigned to train/validate/test:\n"
            f"{badRows.to_string(index=False)}"
        )

    return merged


def validateBasketSplitConsistency(clicks: pd.DataFrame, dataset: str):
    splitCounts = (
        clicks.groupby(["user_id", "time_int"])["split_name"]
        .nunique()
        .reset_index(name="num_splits")
    )

    mixed = splitCounts.loc[splitCounts["num_splits"] > 1]
    if len(mixed) > 0:
        badKeys = mixed.head(10)
        raise ValueError(
            f"{dataset}: some reconstructed baskets span multiple splits, "
            f"so basket boundaries cannot be recovered cleanly.\n"
            f"{badKeys.to_string(index=False)}"
        )


def buildBasketSplits(clicks: pd.DataFrame):
    basketDf = (
        clicks.groupby(["user_id", "time_int", "split_name"], sort=False)
        .agg(
            basket=("item_id", lambda xs: [int(x) for x in xs.tolist()]),
            first_seq_order=("seq_order", "min"),
        )
        .reset_index()
    )

    basketDf["timestamp"] = pd.to_datetime(basketDf["time_int"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")

    splitOrder = {"train": 0, "validate": 1, "test": 2}
    basketDf["split_order"] = basketDf["split_name"].map(splitOrder)

    basketDf = basketDf.sort_values(
        ["user_id", "time_int", "split_order", "first_seq_order"],
        kind="stable",
    ).reset_index(drop=True)

    return basketDf


def validateUserSplitShape(basketDf: pd.DataFrame, dataset: str):
    counts = basketDf.groupby(["user_id", "split_name"]).size().unstack(fill_value=0)

    if "train" in counts:
        trainCounts = counts["train"]
    else:
        trainCounts = pd.Series(0, index=counts.index, dtype=int)

    if "validate" in counts:
        validateCounts = counts["validate"]
    else:
        validateCounts = pd.Series(0, index=counts.index, dtype=int)

    if "test" in counts:
        testCounts = counts["test"]
    else:
        testCounts = pd.Series(0, index=counts.index, dtype=int)

    totalTrain = int(trainCounts.sum())
    totalValidate = int(validateCounts.sum())
    totalTest = int(testCounts.sum())

    if totalTrain == 0:
        raise ValueError(f"{dataset}: no train baskets were reconstructed.")

    if totalValidate == 0:
        raise ValueError(f"{dataset}: no validation baskets were reconstructed.")

    if totalTest == 0:
        raise ValueError(f"{dataset}: no test baskets were reconstructed.")

    trainOnlyMask = (validateCounts == 0) & (testCounts == 0)
    numTrainOnlyUsers = int(trainOnlyMask.sum())

    if numTrainOnlyUsers > 0:
        print(
            f"{dataset}: warning: {numTrainOnlyUsers} users have no validation/test basket "
            f"after conversion; keeping them as train-only users."
        )

def maybeLinkRawDir(outputRoot: Path, dataset: str, rawSourceRoot: Path | None, useSymlink: bool):
    if rawSourceRoot is None:
        return

    sourceRawDir = rawSourceRoot / dataset / "raw"
    targetRawDir = outputRoot / dataset / "raw"

    if not sourceRawDir.exists():
        return

    targetRawDir.parent.mkdir(parents=True, exist_ok=True)

    if targetRawDir.exists() or targetRawDir.is_symlink():
        return

    linkOrCopy(sourceRawDir, targetRawDir, useSymlink)


def writeTifuFiles(outputRoot: Path, dataset: str, basketDf: pd.DataFrame):
    splitDir = outputRoot / dataset / "split"
    splitDir.mkdir(parents=True, exist_ok=True)

    users = sorted(int(x) for x in basketDf["user_id"].unique().tolist())
    userMapping = pd.DataFrame({
        "index": list(range(len(users))),
        "user_id": users,
    })
    user2index = dict(zip(userMapping["user_id"], userMapping["index"]))

    uniqueItems = sorted({
        int(itemId)
        for basket in basketDf["basket"].tolist()
        for itemId in basket
    })
    itemMapping = pd.DataFrame({
        "index": list(range(len(uniqueItems))),
        "item_id": uniqueItems,
    })
    item2index = dict(zip(itemMapping["item_id"], itemMapping["index"]))

    mapped = basketDf.copy()
    mapped["user_id"] = mapped["user_id"].map(user2index).astype(int)
    mapped["basket"] = mapped["basket"].apply(
        lambda basket: [int(item2index[int(itemId)]) for itemId in basket]
    )

    trainDf = (
        mapped.loc[mapped["split_name"] == "train", ["user_id", "timestamp", "basket"]]
        .reset_index(drop=True)
    )
    validateDf = (
        mapped.loc[mapped["split_name"] == "validate", ["user_id", "timestamp", "basket"]]
        .reset_index(drop=True)
    )
    testDf = (
        mapped.loc[mapped["split_name"] == "test", ["user_id", "timestamp", "basket"]]
        .reset_index(drop=True)
    )

    userMapping.to_csv(splitDir / "user_mapping.csv", index=False)
    itemMapping.to_csv(splitDir / "item_mapping.csv", index=False)
    trainDf.to_csv(splitDir / "train.csv", index=False)
    validateDf.to_csv(splitDir / "validate.csv", index=False)
    testDf.to_csv(splitDir / "test.csv", index=False)

    return splitDir, trainDf, validateDf, testDf


def convertTaiwProfileToTifu(taiwRoot: Path, outputRoot: Path, dataset: str, rawSourceRoot: Path | None, useSymlink: bool):
    bookDf, trainDf, devDf, testDf = loadTaiwFiles(taiwRoot, dataset)

    clicks = buildClicksFromBook(bookDf)
    assignments = buildSplitAssignments(trainDf, devDf, testDf)
    clicks = attachSplits(clicks, assignments)

    validateBasketSplitConsistency(clicks, dataset)

    basketDf = buildBasketSplits(clicks)
    validateUserSplitShape(basketDf, dataset)

    splitDir, trainOut, validateOut, testOut = writeTifuFiles(outputRoot, dataset, basketDf)
    maybeLinkRawDir(outputRoot, dataset, rawSourceRoot, useSymlink)

    summary = {
        "dataset": dataset,
        "numUsers": int(basketDf["user_id"].nunique()),
        "trainRows": len(trainOut),
        "validateRows": len(validateOut),
        "testRows": len(testOut),
        "splitDir": str(splitDir),
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["dunnhumby", "tafeng"])
    parser.add_argument(
        "--taiw-root",
        default="/notebooks/A2FYP/hybrid_nbr/data/taiw_profile",
    )
    parser.add_argument(
        "--output-root",
        default="/notebooks/A2FYP/hybrid_nbr/data/taiw_profile_tifu",
    )
    parser.add_argument(
        "--raw-source-root",
        default="/notebooks/A2FYP/time_dependent_nbr/data",
    )
    parser.add_argument("--no-symlink", action="store_true")
    args = parser.parse_args()

    rawSourceRoot = Path(args.raw_source_root) if args.raw_source_root else None

    summary = convertTaiwProfileToTifu(
        taiwRoot=Path(args.taiw_root),
        outputRoot=Path(args.output_root),
        dataset=args.dataset,
        rawSourceRoot=rawSourceRoot,
        useSymlink=not args.no_symlink,
    )

    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()