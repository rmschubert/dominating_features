import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_MNIST, load_torch_breastcancer, load_torch_iris

sns.set_theme()
sns.set_context("talk")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.weight"] = "bold"


Path("results", "plots").mkdir(parents=True, exist_ok=True)

datasets = {
    "MNIST": load_MNIST,
    "Iris": load_torch_iris,
    "Breastcancer": load_torch_breastcancer,
}

factors = ["0.0", "0.001", "0.01", "0.1", "1.0"]

for name, load_fun in datasets.items():

    print(f"Processing {name} ...")

    fig, ax = plt.subplots(1, 3, figsize=(15, 7))

    dataset = load_fun()
    X = dataset.tensors[0]
    y = dataset.tensors[1]

    frequencies = np.zeros(shape=(len(factors) * 5, X.shape[-1]))
    dominance = np.zeros_like(frequencies)

    factors_and_splits = []

    for f, factor in enumerate(factors):
        print(f"for factor {factor}")
        factor_split = factor.split(".")
        factor_pattern = factor_split[0] + r"\." + factor_split[1]
        pattern = name + r".*" + factor_pattern + r"_"
        models = [
            path
            for path in list(Path("results", "models").glob("*.pt"))
            if re.search(pattern, str(path))
        ]

        print(models)

        for j, model_path in enumerate(models):
            glvq_model = torch.load(model_path, weights_only=False)
            split_label = re.search(r"split[0-9]", str(model_path)).group()

            # get the distances
            _, glvq_distances = glvq_model.winner_assignment(X)

            # sort the distances in ascending order to easily access winner and runner up
            sorted_glvq_distances, sorted_glvq_indices = torch.sort(glvq_distances)

            new_assignments = []

            index_closest, index_runner_up = (
                sorted_glvq_indices[:, 0],
                sorted_glvq_indices[:, 1],
            )

            # competing prototypes
            prototype_closest, prototype_runner_up = (
                glvq_model.prototypes[index_closest, :],
                glvq_model.prototypes[index_runner_up, :],
            )

            # classes of the competing prototypes
            class_closest, class_runner_up = (
                glvq_model.prototype_classes[index_closest],
                glvq_model.prototype_classes[index_runner_up],
            )

            # distances of the competing prototypes
            distance_closest, distance_runner_up = (
                sorted_glvq_distances[:, 0],
                sorted_glvq_distances[:, 1],
            )

            # d(x, closest) - d(x, runner_up)
            original_distance_comparison = distance_closest - distance_runner_up

            # the sum terms of the distances -> d(x, closest)_i and d(x, runner_up)_i
            sum_terms_closest = torch.abs(X - prototype_closest) ** 2
            sum_terms_runner_up = torch.abs(X - prototype_runner_up) ** 2

            # the difference between the sum terms
            # delta_i = d(x_i, closest_i) - d(x_i, runner_up_i)
            difference_of_sum_terms = sum_terms_closest - sum_terms_runner_up

            ratio = difference_of_sum_terms / original_distance_comparison.unsqueeze(1)
            dominance[5 * f + j] = ratio.median(dim=0)[0].detach().numpy()
            factors_and_splits.append((factor, split_label))

            sorted_ratio, _ = torch.sort(ratio, descending=True)

            # compute cardinality of dominating features
            n_samples = []
            # c = 1
            remaining_samples_0 = torch.ones(len(X)).bool()
            remaining_samples_t = remaining_samples_0

            c = (torch.cumsum(sorted_ratio, -1) >= 1).int().argmax(-1) + 1
            cardinalities, counts = torch.unique(c, return_counts=True)
            all_cardinalities = torch.arange(1, X.shape[-1] + 1, 1)
            total_counts = torch.tensor(
                [
                    (
                        counts[cardinalities.tolist().index(cardinality)]
                        if cardinality in cardinalities
                        else 0
                    )
                    for cardinality in all_cardinalities
                ]
            )
            relative_frequency = total_counts / len(X)

            frequencies[5 * f + j] = relative_frequency.numpy()

    factors_and_splits_df = pd.DataFrame(
        factors_and_splits, columns=["factors", "splits"]
    )
    dominance_df = pd.DataFrame(
        dominance, columns=[str(i + 1) for i in range(dominance.shape[-1])]
    )
    dominance_df = pd.concat((factors_and_splits_df, dominance_df), axis=1)
    dominance_df = dominance_df.melt(
        ["factors", "splits"], var_name="feature", value_name="relative dominance"
    )
    sns.lineplot(
        data=dominance_df,
        x="feature",
        y="relative dominance",
        ax=ax[0],
        legend=False,
        hue="factors",
        palette=sns.color_palette("colorblind", n_colors=len(factors)),
    )
    dominance_df.to_csv(Path("results", name + "_domination.csv"))

    frequency_df = pd.DataFrame(
        frequencies, columns=[str(i + 1) for i in range(frequencies.shape[-1])]
    )
    frequency_df = pd.concat((factors_and_splits_df, frequency_df), axis=1)
    frequency_df = frequency_df.melt(
        ["factors", "splits"],
        var_name="cardinality",
        value_name="relative frequency",
    )
    sns.lineplot(
        data=frequency_df,
        x="cardinality",
        y="relative frequency",
        ax=ax[1],
        hue="factors",
        palette=sns.color_palette("colorblind", n_colors=len(factors)),
    )
    frequency_df.to_csv(Path("results", name + "_frequencies.csv"))

    if X.shape[-1] < 10:
        xticks = range(X.shape[-1])
    else:
        spacing = X.shape[-1] // 10
        xticks = [i * spacing for i in range(10)]

    ax[0].set_xticks(xticks)
    ax[0].set_xticklabels([str(i + 1) for i in xticks])
    ax[0].tick_params(axis="x", which="major", labelsize=10, rotation=45)
    ax[0].tick_params(axis="y", which="major", labelsize=10)
    ax[0].set_ylabel("relative domination", fontsize=15)
    ax[0].set_xlabel("feature", fontsize=15)
    ax[1].set_xticks(xticks)
    ax[1].set_xticklabels([str(i + 1) for i in xticks])
    ax[1].tick_params(axis="x", which="major", labelsize=10, rotation=45)
    ax[1].tick_params(axis="y", which="major", labelsize=10)
    ax[1].set_ylabel("relative frequency", fontsize=15)
    ax[1].set_xlabel("cardinality", fontsize=15)
    ax[1].legend()

    accuracy_df = pd.read_csv(Path("results", name + "_results.csv"))
    sns.stripplot(
        data=accuracy_df,
        x="factor",
        y="accuracy",
        ax=ax[2],
        hue="factor",
        dodge=True,
        alpha=0.85,
        legend=False,
        palette=sns.color_palette("colorblind", n_colors=len(factors)),
    )
    sns.pointplot(
        data=accuracy_df,
        x="factor",
        y="accuracy",
        ax=ax[2],
        hue="factor",
        dodge=0.4,
        linestyle="none",
        errorbar="sd",
        marker=".",
        markersize=15,
        markeredgewidth=1.5,
        alpha=0.5,
        capsize=0.3,
        palette=sns.color_palette("colorblind", n_colors=len(factors)),
        legend=False,
    )

    ax[2].tick_params(axis="x", which="major", labelsize=10, rotation=45)
    ax[2].tick_params(axis="y", which="major", labelsize=10)
    ax[2].set_ylabel("test accuracy", fontsize=15)
    ax[2].set_xlabel("factor", fontsize=15)
    fig.tight_layout()

    plt.savefig(Path("results", "plots", name + "_cardinality_plot.pdf"), dpi=350)
    plt.show()
    plt.close()
