from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def buildDataFrame():
    rows = [
        {"dataset": "Dunnhumby", "modelFamily": "TIFU", "profile": "TIFU", "configuration": "TIFU on TIFU", "precisionAt10": 0.119339, "recallAt10": 0.269621, "ndcgAt10": 0.236005},
        {"dataset": "Dunnhumby", "modelFamily": "TAIW", "profile": "TAIW", "configuration": "TAIW on TAIW", "precisionAt10": 0.121006, "recallAt10": 0.176589, "ndcgAt10": 0.170829},
        {"dataset": "Dunnhumby", "modelFamily": "TAIW", "profile": "TIFU", "configuration": "TAIW on TIFU", "precisionAt10": 0.122402, "recallAt10": 0.239147, "ndcgAt10": 0.217033},
        {"dataset": "Dunnhumby", "modelFamily": "TIFU", "profile": "TAIW", "configuration": "TIFU on TAIW", "precisionAt10": 0.117947, "recallAt10": 0.216829, "ndcgAt10": 0.203280},
        {"dataset": "TaFeng", "modelFamily": "TIFU", "profile": "TIFU", "configuration": "TIFU on TIFU", "precisionAt10": 0.066843, "recallAt10": 0.167099, "ndcgAt10": 0.145287},
        {"dataset": "TaFeng", "modelFamily": "TAIW", "profile": "TAIW", "configuration": "TAIW on TAIW", "precisionAt10": 0.064566, "recallAt10": 0.157389, "ndcgAt10": 0.127983},
        {"dataset": "TaFeng", "modelFamily": "TAIW", "profile": "TIFU", "configuration": "TAIW on TIFU", "precisionAt10": 0.061626, "recallAt10": 0.136862, "ndcgAt10": 0.112779},
        {"dataset": "TaFeng", "modelFamily": "TIFU", "profile": "TAIW", "configuration": "TIFU on TAIW", "precisionAt10": 0.063871, "recallAt10": 0.164065, "ndcgAt10": 0.147858},
    ]
    return pd.DataFrame(rows)

def ensureOutputDir():
    outputDir = Path("/notebooks/A2FYP/hybrid_nbr/results/baseline_plots")
    outputDir.mkdir(parents=True, exist_ok=True)
    return outputDir

def plotGroupedBar(df, metricColumn, metricLabel, outputPath):
    order = ["TIFU on TIFU", "TAIW on TAIW", "TAIW on TIFU", "TIFU on TAIW"]
    datasets = ["Dunnhumby", "TaFeng"]
    colours = {
        "Dunnhumby": "#4C72B0",
        "TaFeng": "#DD8452",
    }
    hatches = {
        "Dunnhumby": "///",
        "TaFeng": "\\\\\\",
    }

    x = list(range(len(order)))
    barWidth = 0.36

    plt.figure(figsize=(10, 6))

    for index, dataset in enumerate(datasets):
        subset = df[df["dataset"] == dataset].copy()
        subset["configuration"] = pd.Categorical(subset["configuration"], categories=order, ordered=True)
        subset = subset.sort_values("configuration")
        offset = -barWidth / 2 if index == 0 else barWidth / 2
        positions = [value + offset for value in x]

        bars = plt.bar(
            positions,
            subset[metricColumn].tolist(),
            width=barWidth,
            label=dataset,
            color=colours[dataset],
            edgecolor="black",
            linewidth=1.0
        )

        for bar in bars:
            bar.set_hatch(hatches[dataset])

    plt.xticks(x, order, rotation=20, ha="right")
    plt.ylabel(metricLabel)
    plt.xlabel("Baseline configuration")
    plt.title(f"{metricLabel} across baseline configurations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outputPath, dpi=300, bbox_inches="tight")
    plt.close()

def plotProfileSwitchSensitivity(df, outputPath):
    markerMap = {
        ("TIFU", "Dunnhumby"): "o",
        ("TIFU", "TaFeng"): "s",
        ("TAIW", "Dunnhumby"): "^",
        ("TAIW", "TaFeng"): "D",
    }
    colourMap = {
        "Dunnhumby": "#4C72B0",
        "TaFeng": "#DD8452",
    }

    plt.figure(figsize=(10, 6))
    xPositions = [0, 1]
    xLabels = ["Native profile", "Switched profile"]

    for modelFamily in ["TIFU", "TAIW"]:
        for dataset in ["Dunnhumby", "TaFeng"]:
            nativeProfile = modelFamily
            subset = df[(df["modelFamily"] == modelFamily) & (df["dataset"] == dataset)].copy()
            nativeRow = subset[subset["profile"] == nativeProfile].iloc[0]
            switchedRow = subset[subset["profile"] != nativeProfile].iloc[0]

            nativeMean = (nativeRow["precisionAt10"] + nativeRow["recallAt10"] + nativeRow["ndcgAt10"]) / 3
            switchedMean = (switchedRow["precisionAt10"] + switchedRow["recallAt10"] + switchedRow["ndcgAt10"]) / 3

            plt.plot(
                xPositions,
                [nativeMean, switchedMean],
                marker=markerMap[(modelFamily, dataset)],
                markersize=8,
                linewidth=2,
                color=colourMap[dataset],
                label=f"{modelFamily} on {dataset}",
            )

    handles = []
    labels = []
    seen = set()
    for line in plt.gca().get_lines():
        label = line.get_label()
        if label not in seen:
            handles.append(line)
            labels.append(label)
            seen.add(label)

    plt.xticks(xPositions, xLabels)
    plt.ylabel("Mean of Precision@10, Recall@10, and NDCG@10")
    plt.xlabel("Profile setting")
    plt.title("Sensitivity to preprocessing profile switch")
    plt.legend(handles, labels)
    plt.tight_layout()
    plt.savefig(outputPath, dpi=300, bbox_inches="tight")
    plt.close()

def plotCompactMatrix(df, outputPath):
    order = ["TIFU on TIFU", "TAIW on TAIW", "TAIW on TIFU", "TIFU on TAIW"]
    datasets = ["Dunnhumby", "TaFeng"]
    markers = {
        "precisionAt10": "o",
        "recallAt10": "s",
        "ndcgAt10": "^",
    }
    colours = {
        "Dunnhumby": "#4C72B0",
        "TaFeng": "#DD8452",
    }
    metricLabels = {
        "precisionAt10": "Precision@10",
        "recallAt10": "Recall@10",
        "ndcgAt10": "NDCG@10",
    }

    plt.figure(figsize=(11, 6))

    for dataset in datasets:
        subset = df[df["dataset"] == dataset].copy()
        subset["configuration"] = pd.Categorical(subset["configuration"], categories=order, ordered=True)
        subset = subset.sort_values("configuration")

        for metricColumn, marker in markers.items():
            xValues = list(range(len(order)))
            yValues = subset[metricColumn].tolist()
            plt.plot(
                xValues,
                yValues,
                marker=marker,
                markersize=7,
                linewidth=2,
                color=colours[dataset],
                alpha=0.9,
            )

    datasetHandles = [
        Line2D([0], [0], color=colours["Dunnhumby"], linewidth=2, label="Dunnhumby"),
        Line2D([0], [0], color=colours["TaFeng"], linewidth=2, label="TaFeng"),
    ]
    metricHandles = [
        Line2D([0], [0], color="black", marker=markers[column], linewidth=0, markersize=7, label=label)
        for column, label in metricLabels.items()
    ]

    plt.xticks(list(range(len(order))), order, rotation=20, ha="right")
    plt.ylabel("Metric value")
    plt.xlabel("Baseline configuration")
    plt.title("Compact baseline matrix comparison")
    firstLegend = plt.legend(handles=datasetHandles, loc="upper right")
    plt.gca().add_artist(firstLegend)
    plt.legend(handles=metricHandles, loc="lower left")
    plt.tight_layout()
    plt.savefig(outputPath, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    outputDir = ensureOutputDir()
    df = buildDataFrame()
    df.to_csv(outputDir / "baseline_matrix.csv", index=False)

    plotGroupedBar(df, "precisionAt10", "Precision@10", outputDir / "precision_at10_grouped_bar.png")
    plotGroupedBar(df, "recallAt10", "Recall@10", outputDir / "recall_at10_grouped_bar.png")
    plotGroupedBar(df, "ndcgAt10", "NDCG@10", outputDir / "ndcg_at10_grouped_bar.png")
    plotProfileSwitchSensitivity(df, outputDir / "profile_switch_sensitivity.png")
    plotCompactMatrix(df, outputDir / "compact_baseline_matrix.png")

    print(f"Saved plots to: {outputDir}")

if __name__ == "__main__":
    main()