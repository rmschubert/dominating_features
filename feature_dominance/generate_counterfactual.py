import torch


def create_counterfactual_incremental(model, x, target_class, criterion: float = 1e-1):
    prototypes = model.prototypes.detach()
    target_prototype = prototypes[target_class]

    counterfactual = torch.clone(x)
    dominating_features = torch.zeros(x.shape).bool()
    minimal_cardinality_0 = 0
    minimal_cardinality_t = minimal_cardinality_0
    sorted_inds = torch.ones(x.shape)
    rel_sim = 0
    trajectory = []
    class_history = []

    while True:
        winner, distances = model.winner_assignment(counterfactual)
        sorted_distances, _ = torch.sort(distances)
        margin = 0.5 * (sorted_distances[1] - sorted_distances[0])
        if winner == target_class:
            if margin > criterion:
                rel_sim = 2 * margin / (sorted_distances[0] + sorted_distances[1])
                break
            else:
                minimal_cardinality_t = minimal_cardinality_t + 1
                dominating_subset = sorted_inds[:minimal_cardinality_t]
                dominating_features[dominating_subset] = True
                counterfactual = torch.where(dominating_features, target_prototype, x)
                trajectory.append(counterfactual)
                class_history.append(winner)

                if all(dominating_features):
                    rel_sim = 2 * margin / (sorted_distances[0] + sorted_distances[1])
                    break
        else:
            absolute_difference = torch.abs(prototypes - counterfactual) ** 2

            winner_difference = absolute_difference[winner]
            target_difference = absolute_difference[target_class]
            competing_terms = winner_difference - target_difference

            relative_dominance = competing_terms / (competing_terms.sum())
            sorted_dominance, sorted_inds = torch.sort(
                relative_dominance, descending=True
            )
            minimal_cardinality_t = (
                (torch.cumsum(sorted_dominance, 0) > 1).int().argmax()
            )
            if (competing_terms.sum()) >= 0:
                print(minimal_cardinality_t, (competing_terms.sum()), winner, distances)
            dominating_subset = sorted_inds[: minimal_cardinality_t + 1]
            dominating_features[dominating_subset] = True
            counterfactual = torch.where(dominating_features, target_prototype, x)
            trajectory.append(counterfactual)
            class_history.append(winner)

    return counterfactual, dominating_features, trajectory, class_history, rel_sim
