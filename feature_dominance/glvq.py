import torch
from torch import Tensor
from torch.nn import Module, Parameter, ReLU


class GLVQ(Module):

    def __init__(
        self,
        prototypes: Tensor,
        class_distribution: list[int] | tuple[int, ...],
        representational_factor: float = 0.0,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.device = torch.device(device)

        self.prototype_distribution = torch.tensor(class_distribution).to(self.device)

        prototype_classes = [
            cls
            for cls, repetition in enumerate(class_distribution)
            for _ in range(repetition)
        ]
        self.register_buffer(
            "prototype_classes",
            torch.tensor(prototype_classes),
        )

        self.prototypes = Parameter(prototypes)

        self.register_buffer(
            "representational_factor",
            torch.tensor(representational_factor),
        )

        # we use ReLU activation
        self.stabilizer = ReLU()
        self.activation = self.stabilizer

        self.to(device)

    def belongingness(self, x: Tensor, prototypes: Tensor) -> Tensor:

        xx = torch.sum(x**2, dim=-1, keepdim=True)
        xy = -2 * x @ prototypes.T
        yy = torch.sum(prototypes**2, dim=-1)

        # the above can become negative, due to numerical instabilities,
        # thus we need to clip using e.g. ReLU
        distances = self.stabilizer(xx + xy + yy)
        return distances

    def imposer(self, label: Tensor) -> tuple[Tensor, Tensor]:
        # determine the masks to find closest prototype from same class
        # and closest prototype from another class
        prototype_label = self.prototype_classes
        prototype_label, label = torch.broadcast_tensors(
            prototype_label.unsqueeze(0), label.unsqueeze(1)
        )
        differences = prototype_label - label
        matcher = differences == 0
        not_matcher = ~matcher

        return matcher, not_matcher

    def forward(self, x: Tensor, label: Tensor) -> tuple[Tensor, ...]:
        prototypes = self.prototypes

        # calculate distances
        belongingness = self.belongingness(x, prototypes)

        # determine the masks
        matcher, not_matcher = self.imposer(label)

        # filter for matching and mismatching classes
        filtered_correct = belongingness[matcher]
        filtered_incorrect = belongingness[not_matcher]

        # filter the distances as well
        prototypes_per_class = self.prototype_distribution.unique().item()
        remaining_prototypes = len(self.prototype_classes) - prototypes_per_class

        dp, _ = torch.min(filtered_correct.reshape(len(label), prototypes_per_class), 1)

        dm, _ = torch.min(
            filtered_incorrect.reshape(len(label), remaining_prototypes), 1
        )

        # classification function
        mu = (dp - dm) / (dp + dm)

        # local errors with representation term
        local_errors = self.activation(mu) + self.representational_factor * dp

        return local_errors

    def winner_assignment(self, x: Tensor) -> Tensor:
        # nearest prototype principle
        prototypes = self.prototypes

        belongingness = self.belongingness(x, prototypes)

        winner = torch.argmin(belongingness, -1)
        assignment = self.prototype_classes[winner]

        return assignment, belongingness
