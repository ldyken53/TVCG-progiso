#!/usr/bin/env python3

import sys
import os
import re
import json
import numpy as np
import matplotlib
from matplotlib import rc
from matplotlib import cm
import matplotlib.pyplot as plt

# Each data set has its list of performance at the given resolutions,
# ordered as in the resolution map above. Key's are the dataset name
image_completeness = [
    0.85,
    1
]
prog_iso_scaling = {k: {} for k in image_completeness}
bcmc_perf = {}
dataset_sizes = {}

dataset_labels = {
    "magnetic_reconnection": "Plasma",
    "chameleon": "Chameleon",
    "miranda": "Miranda"
}

match_benchmark_name = re.compile("(\\w+)\\/perf-(\\w+)_(\\d+x\\d+x\\d+)_\\w+.raw.crate\\d-[a-z]+-[a-z]+-([0-9]+)p-([0-9])ssc-([0-9]\.*[0-9]*)r.*\\.json")
match_benchmark_name2 = re.compile("(\\w+)\\/perf-(\\w+)_(\\d+x\\d+x\\d+)_\\w+.raw.crate\\d-[a-z]+-[a-z]+-([0-9]+)p-([0-9])ssc-.*\\.json")

# Compute and plot utilization statistics for the benchmarks
directory = "benchmarks"
for results_file in os.listdir(directory):
    file_name = os.path.join(directory, results_file)
    m = match_benchmark_name.search(file_name)
    if not m:
        m = match_benchmark_name2.search(file_name)
    if m:
        print(f"Matched {file_name}, dataset: {m.group(1)}, dims: {m.group(4)}")
        dataset = m.group(2)
        spec_count = m.group(5)
        if dataset not in dataset_labels.keys():
            continue
        dataset_sizes[dataset] = m.group(2)
        if not dataset in prog_iso_scaling[1]:
            for k in image_completeness:
                prog_iso_scaling[k][dataset] = {}

        with open(file_name, "r") as f:
            b = json.load(f)
            time_to_complete = []
            img_size = b[0]["stats"][0]["imageSize"]
            resolution = f"{img_size[0]}x{img_size[1]}"
            if not resolution in prog_iso_scaling[1][dataset]:
                for k in image_completeness:
                    prog_iso_scaling[k][dataset][resolution] = []

            for run in b:
                total_time = {k: 0 for k in image_completeness}
                estimate_reached = {k: False for k in image_completeness}
                for s in run["stats"]:
                    for k in image_completeness:
                        if not estimate_reached[k]:
                            total_time[k] += s["totalPassTime_ms"]
                        if s["imageCompleteness"] > k:
                            estimate_reached[k] = True

                time_to_complete.append(total_time)

            for k in image_completeness:
                prog_iso_scaling[k][dataset][resolution].append(np.mean([pas[k] for pas in time_to_complete]))
            
            print(f"""Results {dataset} @ {img_size[0]}x{img_size[1]}:
            Avg. Total Time: {np.mean([pas[k] for pas in time_to_complete]):.2f}ms
            Starting Speculation Count: {spec_count}
            """)
    else:
        print(f"Failed to match {file_name}")


resolution_order = {
    "640x360": 0,
    "1280x720": 1,
    "1920x1080": 2
}
colors = {
    "640x360": "tab:blue",
    "1280x720": "tab:orange",
    "1920x1080": "tab:green"
}

for k in image_completeness:
    fig, ax = plt.subplots()
    # Build the plot sets for the bars to have the right labels/grouping
    resolution_results = {
        "640x360": [],
        "1280x720": [],
        "1920x1080": []
    }
    max_y = 0
    idx = 0
    for name in dataset_labels.keys():
        for res in resolution_order.keys():
            if prog_iso_scaling[k][name].get(res):
                prog_iso_scaling[k][name][res].sort()

                # We need to plot each bar on its own to get the coloring by resolution
                resolution_results[res].append(prog_iso_scaling[k][name][res][0])

                max_y = max(max_y, prog_iso_scaling[k][name][res][0])
                idx += 1
            else:
                resolution_results[res].append(0)
                idx += 1


    group_width = 0.5
    offset = -group_width / 3.0
    x_ticks = np.arange(len(resolution_results["1280x720"]))
    for res, results in resolution_results.items():
        ax.bar(x_ticks + offset, results, group_width / 3, label=res, color=colors[res])
        offset += group_width / 3.0

    benchmark_names = [dataset_labels[n] for n in dataset_labels.keys()]

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.xaxis.set_ticks(x_ticks)
    plt.xticks(x_ticks, benchmark_names)
    ax.set_ylabel("Avg. Total Time (ms)")


    ax.set_xlabel("Data set")
    ax.set_ylim((0, max_y + max_y * .1))

    plt.legend(loc="upper left")
    plt.savefig(f"ResultsAt{int(k * 100)}%Complete.png", dpi=300, bbox_inches="tight")

