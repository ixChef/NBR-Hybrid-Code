import argparse
import shutil
import sys
from pathlib import Path

projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot))

from configs.profiles import taiwProfile

repoRoot = Path(taiwProfile["repoRoot"])
sys.path.insert(0, str(repoRoot))

from nbr.preparation.preprocess import Preprocess, save_split

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
    sourceFile = sourceDataRoot / f"{datasetName}.txt"
    targetRawDir = outputRoot / "raw"
    targetRawDir.mkdir(parents=True, exist_ok=True)
    targetFile = targetRawDir / f"{datasetName}.txt"
    if not sourceFile.exists():
        raise FileNotFoundError(f"Missing source raw file: {sourceFile}")
    linkOrCopy(sourceFile, targetFile, profile["useSymlink"])
    return targetRawDir

def clearDatasetOutput(profile: dict, datasetName: str):
    dataDir = Path(profile["outputRoot"]) / f"data_{datasetName}"
    if dataDir.exists():
        shutil.rmtree(dataDir)

def runTaiwProfile(datasetName: str, force: bool = False):
    if datasetName not in taiwProfile["datasets"]:
        raise ValueError(f"Unsupported dataset: {datasetName}")

    if force:
        clearDatasetOutput(taiwProfile, datasetName)

    rawDir = prepareRawData(taiwProfile, datasetName)

    outputRoot = Path(taiwProfile["outputRoot"])
    outputRoot.mkdir(parents=True, exist_ok=True)

    corpus = Preprocess(path=str(rawDir) + "/", dataset=datasetName)
    corpus.load_data(
        user_min=taiwProfile["userMin"],
        item_min=taiwProfile["itemMin"],
        filt=taiwProfile["applyNoiseUserFilter"]
    )
    save_split(str(outputRoot), datasetName, corpus)

    dataDir = outputRoot / f"data_{datasetName}"
    summary = {
        "dataset": datasetName,
        "profile": taiwProfile["profileName"],
        "numUsers": corpus.n_users,
        "numItems": corpus.n_items,
        "numClicks": corpus.n_clicks,
        "outputDir": str(dataDir)
    }
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(taiwProfile["datasets"].keys()))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    summary = runTaiwProfile(
        datasetName=args.dataset,
        force=args.force
    )

    for key, value in summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()