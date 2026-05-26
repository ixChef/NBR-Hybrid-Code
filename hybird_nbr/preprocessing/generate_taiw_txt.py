import argparse
from pathlib import Path

import pandas as pd


projectRoot = Path(__file__).resolve().parents[1]
workspaceRoot = projectRoot.parent


def buildDunnhumby():
    rawPath = workspaceRoot / "time_dependent_nbr" / "data" / "dunnhumby" / "raw" / "transaction_data.csv"
    outPath = workspaceRoot / "time_aware_item_weighting" / "data" / "dunnhumby.txt"

    df = pd.read_csv(rawPath)

    df = df.rename(columns={
        "household_key": "user_name",
        "PRODUCT_ID": "item_name",
    })

    # same time logic as the TIFU-side Dunnhumby loader
    timestamp = pd.to_datetime(
        df["DAY"] * 1440 + (df["TRANS_TIME"] // 100) * 60 + (df["TRANS_TIME"] % 100),
        unit="m",
    )

    df["time"] = (timestamp.astype("int64") // 10**9).astype(int)

    out = df[["user_name", "item_name", "time"]].drop_duplicates()
    outPath.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outPath, sep="\t", index=False, header=False)

    return outPath, len(out)


def buildTafeng():
    rawPath = workspaceRoot / "time_dependent_nbr" / "data" / "tafeng" / "raw" / "ta_feng_all_months_merged.csv"
    outPath = workspaceRoot / "time_aware_item_weighting" / "data" / "tafeng.txt"

    df = pd.read_csv(rawPath)

    df = df.rename(columns={
        "CUSTOMER_ID": "user_name",
        "PRODUCT_ID": "item_name",
    })

    timestamp = pd.to_datetime(df["TRANSACTION_DT"])
    df["time"] = (timestamp.astype("int64") // 10**9).astype(int)

    out = df[["user_name", "item_name", "time"]].drop_duplicates()
    outPath.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outPath, sep="\t", index=False, header=False)

    return outPath, len(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["dunnhumby", "tafeng", "all"])
    args = parser.parse_args()

    builders = {
        "dunnhumby": buildDunnhumby,
        "tafeng": buildTafeng,
    }

    if args.dataset == "all":
        names = ["dunnhumby", "tafeng"]
    else:
        names = [args.dataset]

    for name in names:
        outPath, rows = builders[name]()
        print(f"{name}: wrote {rows} rows to {outPath}")


if __name__ == "__main__":
    main()