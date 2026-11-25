import re
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from datasets import load_MNIST
from generate_counterfactual import create_counterfactual_incremental

target_class = 5

dataset = load_MNIST()
name = "MNIST"
X, y = dataset.tensors
# 6
class_to_investigate = 6
class_mask = y == class_to_investigate
# 12
img_idx = class_mask.nonzero()[12].squeeze()

factors = ["0.0", "0.001", "0.01", "0.1", "1.0"]

fig, ax = plt.subplots(len(factors), 8, figsize=(10, 8))

print("generating counterfactuals ...")
for f, factor in enumerate(factors):
    print(f"for factor {factor}")
    factor_split = factor.split(".")
    factor_pattern = factor_split[0] + r"\." + factor_split[1]
    pattern = name + r".*" + factor_pattern + r"_"
    models = [
        path
        for path in list(sorted(Path("results", "models").glob("*.pt")))
        if re.search(pattern, str(path))
    ]

    sample = X[img_idx]
    ax[f, 0].imshow(sample.view(28, 28).cpu().numpy(), cmap="gray")
    ax[f, 0].axis("off")

    summarized_dominating_features = torch.zeros(784)
    for j, model_path in enumerate(models):
        glvq_model = torch.load(model_path, weights_only=False)
        split_label = re.search(r"split[0-9]", str(model_path)).group()[-1]

        # we can do this since we used only one prototype per class
        target_prototype = glvq_model.prototypes[target_class, :]
        ax[f, 1].imshow(
            target_prototype.view(28, 28).detach().cpu().numpy(), cmap="gray"
        )
        ax[f, 1].axis("off")

        # get the distances
        winner, distances = glvq_model.winner_assignment(X)

        m = torch.logical_and(winner == target_class, winner == y)
        relevant_distances = distances[m]
        sorted_distances, _ = torch.sort(relevant_distances)
        winner_distance, runner_up_distance = (
            sorted_distances[:, 0],
            sorted_distances[:, 1],
        )

        average_signed_margin = torch.mean(0.5 * (runner_up_distance - winner_distance))
        signed_rel_sim = (
            (winner_distance - runner_up_distance)
            / (winner_distance + runner_up_distance)
        ).abs()

        counterfactual, dominating_features, _, _, rel_sim = (
            create_counterfactual_incremental(
                glvq_model, sample, target_class, criterion=average_signed_margin
            )
        )
        confidence = (signed_rel_sim < rel_sim).float().mean()

        ax[f, j + 2].imshow(
            counterfactual.view(28, 28).detach().cpu().numpy(), cmap="gray"
        )
        summarized_dominating_features = (
            summarized_dominating_features + dominating_features
        )
        ax[f, j + 2].set_title(f"{confidence:.3f}")
        ax[f, j + 2].axis("off")

    cb_img = ax[f, -1].imshow(
        (summarized_dominating_features / len(models))
        .view(28, 28)
        .detach()
        .cpu()
        .numpy(),
        cmap="BuPu",
    )
    ax[f, -1].axis("off")
    plt.colorbar(cb_img, ax=ax[f, -1], fraction=0.046, pad=0.04)


plt.subplots_adjust(
    left=0.05, right=0.9, top=0.9, bottom=0.05, wspace=0.05, hspace=-0.4
)
plt.savefig(Path("results", "plots", "counterfactual_example.pdf"), dpi=350)
plt.show()
