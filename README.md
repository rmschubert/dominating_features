# Inspect and Explain: Counterfactual Explanations via Dominating Features

This repository contains code for the experiments and the **CEDS** method described in the paper by Ronny Schubert and Thomas Villmann, “Inspect and Explain: Counterfactual Explanations via Dominating Features in Prototype-based Models.”

The paper combines feature attribution with counterfactual explanations for prototype-based models. It introduces **dominating features** and **decision-dominating feature subsets** as tools for inspecting model decisions, and uses them in **Counterfactual Explanations via Dominating Subsets (CEDS)** to generate counterfactuals.

The idea of dominance is derived from **prototype competition**. A prototype-based classifier compares the distances from a sample to competing prototypes. For distances that can be decomposed feature by feature, the difference between those distances can also be written as a sum of individual feature contributions. Each feature’s relative dominance measures its contribution in relation to the overall competition:

$$
\mathrm{dom}(x_j,k) = \frac{\delta_j}{C_k(x)}.
$$

Here, a positive relative dominance supports the decision being examined, while a negative value opposes it. A decision-dominating subset is a set of supporting features whose contributions, taken together, account for at least the full competition. The method uses such subsets both to inspect which features support a decision and to select features to modify when generating a counterfactual.

The paper demonstrates the approach with Generalized Learning Vector Quantization (GLVQ), a nearest-prototype classifier, and proposes CEDS as a sparse and traceable counterfactual-generation strategy.

**Paper:** [https://doi.org/10.1007/978-3-032-45370-9_5](https://doi.org/10.1007/978-3-032-45370-9_5)  
**Repository:** [rmschubert/dominating_features](https://github.com/rmschubert/dominating_features)

## Citation

If you use this repository or build on the method, please cite the paper:

```bibtex
@InBook{Schubert2026a,
  author    = {Schubert, Ronny and Villmann, Thomas},
  pages     = {48--59},
  publisher = {Springer Nature Switzerland},
  title     = {Inspect and Explain: Counterfactual Explanations via Dominating Features in Prototype-Based Models},
  year      = {2026},
  isbn      = {9783032453709},
  month     = Nov,
  booktitle = {Intelligent Data Engineering and Automated Learning – IDEAL 2026},
  doi       = {10.1007/978-3-032-45370-9_5},
  issn      = {1611-3349},
}
```


## Overview

Prototype-based models classify a sample by comparing its distance to learned prototypes. A feature’s contribution to the competition between the winning prototype and a competing prototype can indicate whether that feature supports or opposes the current decision.

The repository implements experiments using Generalized Learning Vector Quantization (GLVQ), evaluates feature dominance, and demonstrates CEDS counterfactual generation.

## Core definitions

For a sample $x \in \mathbb{R}^n$ and prototype $p \in \mathbb{R}^n$, the experiments use the root-omitted Minkowski distance

$$
d_p(x,p) = \sum_{j=1}^{n} |x_j-p_j|^p,
\qquad 1 \le p < \infty.
$$

For a winner prototype $p^\bullet$ and a competing prototype $p_k$, define their competition as

$$
C_k(x) = d_p(x,p^\bullet)-d_p(x,p^k).
$$

When $p^\bullet$ is the nearest prototype, $C_k(x) \le 0$. The contribution of feature $j$ to this competition is

$$
\delta_j =
|x_j-p^\bullet_{j}|^p - |x_j-p^k_{j}|^p.
$$

The relative dominance of feature $x_j$ is

$$
\mathrm{dom}(x_j,k) =
\frac{\delta_j}{C_k(x)},
\qquad C_k(x) \ne 0.
$$

A **decision-dominating subset** is a subset of features that supports the decision and whose relative dominance sums to at least one:

$$
D_k(x) =
\\{
F \subseteq F(x)
|
\sum_{x_j \in F}\mathrm{dom}(x_j,k) \ge 1
\;\land\;
\mathrm{dom}(x_j,k)>0\ \forall x_j\in F
\\}.
$$

A **minimal dominating subset** is a member of $D_k(x)$ with the smallest cardinality:

$$
F^\ast \in \arg\min_{F \in D_k(x)} |F|.
$$

There may be more than one subset with minimum cardinality.

## CEDS

**Counterfactual Explanations via Dominating Subsets (CEDS)** generates a counterfactual by progressively changing features of an input toward the values of a prototype from the target class. Unlike an approach that searches numerically through arbitrary perturbations, CEDS uses decision-dominating subsets to choose which features to change.

Algorithm outline:

1. **Initialize** the counterfactual candidate $\tilde{x}$ as a copy of the original sample $x$. Choose a target-class prototype $p_t$ and a reliability parameter $\epsilon$.
2. **Check the current prediction.** Find the nearest prototype to $\tilde{x}$ and determine its class. Also identify the nearest competing prototype needed to calculate the hypothesis margin.
3. **If the target class has not been reached,** compute a minimal decision-dominating subset for the competition between the current winner and the target prototype. Add the subset’s features to the set of features to modify, then set those features in $\tilde{x}$ to the corresponding values of $p_t$.
4. **If the target class has been reached but the margin is too small,** select another feature from the remaining features according to its relative dominance, add it to the modified set, and set it to the target prototype’s value. This increases the margin rather than stopping at a counterfactual that may lie very close to the decision boundary.
5. **Repeat** the prediction and margin checks until the target prototype wins and the hypothesis margin is at least $\epsilon$.

For a winner $p^\bullet$ and the nearest competing prototype $p^\ast$, the hypothesis margin is

$$
m(\tilde{x}) =
\frac{d_p(\tilde{x},p^\ast)-d_p(\tilde{x},p^\bullet)}{2}.
$$

Thus, the stopping condition requires both that the target prototype is the winner and that $m(\tilde{x}) \ge \epsilon$. The parameter $\epsilon$ controls the required margin; it must be achievable for the chosen prototypes.

CEDS is described in the paper for a nearest-prototype classifier with one prototype in the target class. With multiple target-class prototypes, the target prototype can be selected before generation or updated as the counterfactual changes. Recording the features added and the intermediate predictions makes the generation process traceable.

## Experiments

The experiments investigate feature dominance and counterfactual generation for GLVQ models:

- **Datasets:** Iris, Breast Cancer Wisconsin Diagnostic, and MNIST.
- **Validation:** Stratified 5-fold cross-validation.
- **Models:** One prototype per class, trained with representation factors $\alpha \in \\{0,10^{-3},10^{-2},10^{-1},1\\}.$
- **Distance:** squared Euclidean distance.
- **Training loss:** ReLU applied to the GLVQ local error, with an additional representation term weighted by $\alpha$.
- **Preprocessing:** Iris and Breast Cancer features were robustly scaled and then min–max normalized to $[0,1]$. MNIST images were flattened.
- **Reported analyses:** Relative dominance, minimal dominating-subset cardinality, test accuracy, counterfactual confidence, and agreement of altered features across folds.

## Counterfactual confidence

We propose a confidence measure based on the _relative similarity_ [1] of correctly classified training samples from the target class. Let $X^+$ be those samples, and let $\mathrm{relsim}$ denote the relative similarity measure used by the model. The counterfactual confidence is

$$
\mathrm{conf}(\tilde{x}) =
\frac{1}{|X^+|}
\sum_{x^j \in X^+}
\mathbf{1}\left(
\mathrm{relsim}(x^j)
<
\mathrm{relsim}(\tilde{x})
\right),
$$

where $\mathbf{1}(\cdot)$ is the indicator function. This score is the average proportion of correctly classified target-class samples with a lower relative-similarity score than the counterfactual.

## Running the code

The paper identifies this repository as the source code for the experiments. Requirements can be installed using the requirements.txt file.

To get started:
   ```bash
   git clone https://github.com/rmschubert/dominating_features.git
   cd dominating_features
   ```

## References
[1] - L. Fischer, B. Hammer, H. Wersing,
Efficient rejection strategies for prototype-based classification,
Neurocomputing,
2015,
https://doi.org/10.1016/j.neucom.2014.10.092.
