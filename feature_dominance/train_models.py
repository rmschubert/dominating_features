from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_MNIST, load_torch_breastcancer, load_torch_iris
from glvq import GLVQ
from sklearn.model_selection import StratifiedKFold
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

results_path = Path("results_tests")
results_and_models_path = Path(results_path, "models")
results_and_models_path.mkdir(parents=True, exist_ok=True)

datasets = {
    "MNIST": load_MNIST,
    "Iris": load_torch_iris,
    "Breastcancer": load_torch_breastcancer,
}

## hyperparameters
batch_size = 32
number_of_epochs = 25

r_terms = [0.0, 1e-3, 1e-2, 1e-1, 1.0]

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

for name, load_fun in datasets.items():
    train_dataset = load_fun()
    X, y = train_dataset.tensors
    accuracies = []

    for s, (train_idx, test_idx) in enumerate(kfold.split(X, y)):

        train_subset = Subset(train_dataset, train_idx)

        test_subset = Subset(train_dataset, test_idx)

        # train and test loader
        train_loader = DataLoader(
            dataset=train_subset, batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            dataset=test_subset, batch_size=len(test_subset), shuffle=True
        )

        # model hyperparameters
        prototype_class_distribution = [
            1 for _ in range(len(train_dataset.tensors[1].unique()))
        ]
        # model loss
        criterion = torch.sum
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # learning rate
        glvq_lr = 1e-1

        ## Gather whole dataset
        train_samples = []
        train_targets = []
        for batch_data, batch_targets in train_loader:
            train_samples.append(batch_data.to(device))
            train_targets.append(batch_targets.to(device))

        x_train = torch.cat(train_samples, dim=0)

        for r in r_terms:
            # initialize the GLVQ instance
            glvq_model = GLVQ(
                prototypes=torch.clone(x_train[: len(prototype_class_distribution)]),
                class_distribution=prototype_class_distribution,
                representational_factor=r,
                device=device,
            )
            # optimizer
            optimizer = Adam(glvq_model.parameters(), lr=glvq_lr)
            scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.75)

            ## Train GLVQ
            print(
                f"Training the GLVQ model with representational term {r} for {name}..."
            )
            for epoch in range(number_of_epochs):
                epoch_train_accuracies = torch.zeros(len(train_loader))
                epoch_test_accuracies = torch.zeros(len(train_loader))
                epoch_losses = torch.zeros(len(train_loader))
                with tqdm(
                    total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{number_of_epochs}",
                    unit="batch",
                    position=0,
                    leave=True,
                ) as pbar:
                    for i, (images, label) in enumerate(train_loader):
                        # make a training step
                        images = images.to(device)
                        label = label.to(device)
                        output = glvq_model(images.detach(), label)
                        loss = criterion(output)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        # constrain prototypes to have values only in [0, 1]
                        glvq_model.prototypes.data = torch.clamp(
                            glvq_model.prototypes.data, 0, 1
                        )

                        prediction, _ = glvq_model.winner_assignment(images)
                        train_accuracy = (prediction == label).to(float).mean()
                        epoch_train_accuracies[i] = train_accuracy

                        # compute the batch accuracy
                        test_batch, test_label = next(iter(test_loader))
                        prediction, _ = glvq_model.winner_assignment(test_batch)
                        test_accuracy = (prediction == test_label).to(float).mean()

                        # epoch_test_accuracies[i] = test_accuracy
                        epoch_losses[i] = loss.item()

                        # register loss and accuracy for epoch in progress bar
                        pbar.set_postfix(
                            {
                                "Loss": epoch_losses.mean().item(),
                                "test_accuracy": epoch_test_accuracies.mean().item(),
                                "train_accuracy": epoch_train_accuracies.mean().item(),
                            }
                        )
                        pbar.update(1)

                    scheduler.step()

            test_batch, test_label = next(iter(test_loader))
            prediction, _ = glvq_model.winner_assignment(test_batch)
            test_accuracy = (prediction == test_label).to(float).mean()
            accuracies.append((r, f"split_{s}", test_accuracy.item()))

            model_name = name + "_repterm_" + f"{r}_split{s}.pt"
            torch.save(glvq_model, Path(results_and_models_path, model_name))
            del glvq_model

            print("Training completed!")

    results_df = pd.DataFrame(accuracies, columns=["factor", "split", "accuracy"])
    results_df.to_csv(Path(results_path, f"{name}_results.csv"))
