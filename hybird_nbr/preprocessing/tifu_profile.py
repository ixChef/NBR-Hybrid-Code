import argparse
import shutil
import sys
from pathlib import Path

projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot))

from configs.profiles import tifuProfile

repoRoot = Path(tifuProfile["repoRoot"])
sys.path.insert(0, str(repoRoot))


import src.dataset.base as tifuBaseModule
from src.dataset.dunnhumby import DunnhumbyDataset
from src.dataset.tafeng import TafengDataset

datasetFactory = {
    "dunnhumby": DunnhumbyDataset,
    "tafeng": TafengDataset
}

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

def prepareRawData(profile: dict, datasetName: str):
    sourceDataRoot = Path(profile["sourceDataRoot"])
    outputRoot = Path(profile["outputRoot"])
    sourceRawDir = sourceDataRoot / datasetName / "raw"
    targetRawDir = outputRoot / datasetName / "raw"
    if not sourceRawDir.exists():
        raise FileNotFoundError(f"Missing source raw directory: {sourceRawDir}")
    targetRawDir.mkdir(parents=True, exist_ok=True)
    for sourcePath in sourceRawDir.iterdir():
        targetPath = targetRawDir / sourcePath.name
        linkOrCopy(sourcePath, targetPath, profile["useSymlink"])
    return targetRawDir

def clearDatasetOutput(profile: dict, datasetName: str):
    datasetDir = Path(profile["outputRoot"]) / datasetName
    if datasetDir.exists():
        shutil.rmtree(datasetDir)

def runTifuProfile(datasetName: str, force: bool = False, verbose: bool = True):
    if datasetName not in datasetFactory:
        raise ValueError(f"Unsupported dataset: {datasetName}")

    if force:
        clearDatasetOutput(tifuProfile, datasetName)

    prepareRawData(tifuProfile, datasetName)

    outputRoot = Path(tifuProfile["outputRoot"])
    outputRoot.mkdir(parents=True, exist_ok=True)

    tifuBaseModule.DATA_DIR = str(outputRoot)

    datasetClass = datasetFactory[datasetName]
    dataset = datasetClass(
        dataset_folder_name=datasetName,
        min_baskets_per_user=tifuProfile["minBasketsPerUser"],
        min_items_per_user=tifuProfile["minItemsPerUser"],
        min_users_per_item=tifuProfile["minUsersPerItem"],
        verbose=verbose
    )

    dataset.preprocess()
    dataset.make_leave_two_baskets_split(
        random_baskets=tifuProfile["randomBaskets"],
        random_state=tifuProfile["randomState"]
    )
    trainDf, valDf, testDf = dataset.load_split()

    splitDir = Path(outputRoot) / datasetName / "split"
    summary = {
        "dataset": datasetName,
        "profile": tifuProfile["profileName"],
        "numUsers": dataset.num_users,
        "numItems": dataset.num_items,
        "trainRows": len(trainDf),
        "validateRows": len(valDf),
        "testRows": len(testDf),
        "splitDir": str(splitDir)
    }
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(datasetFactory.keys()))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    summary = runTifuProfile(
        datasetName=args.dataset,
        force=args.force,
        verbose=not args.quiet
    )

    for key, value in summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()