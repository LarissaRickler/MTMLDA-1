# MTMLDA: Within-Chain Parallelism for Multilevel MCMC

This repository contains a  specialized implementation of the *Multilevel Delayed Acceptance* (MLDA) algorithm. MLDA is a multilevel *Markov Chain Monte Carlo* algorithm, which has been proposed [here](https://doi.org/10.1137/22M1476770). Like other multilevel sampling procedures, MLDA utilizes a hierarchy of models that approximate the target distribution with varying fidelities. The implemented version comprises within-chain parallelism through [prefetching](https://www.tandfonline.com/doi/abs/10.1198/106186006X100579), the expansion of possible future states of the Markov chain in a binary decision tree. The target density evaluations at these states can be performed in parallel, potentially increasing MCMC execution. This is particularly useful for scenarios where burn-in is significant, such that parallelizaion through multiple chains is inefficient. We combine MLDA with asynchronous prefetching, to make full use of a hierarchy of models with potentially vastly different evaluation times. The theoretical background and conceptual setup of the implementation can be found in the accompanying publication, "Scalable Bayesian Inference of Large Simulations via Asynchronous Prefetching Multilevel Delayed Acceptance".

## Installation

The library in this repository is a Python package readily installable via `pip`, simply run
```python
pip install .
```
in the root directory. Alternatively, you may use [UV](https://docs.astral.sh/uv/), which has been used for the setup of the project.

## Usage



## License

This Software is distributed under the [MIT](https://choosealicense.com/licenses/mit/) license.

It has been developed as part of the innovation study **ScalaMIDA**. It has received funding through the **Inno4scale** project, which is funded by the European High-Performance Computing Joint Un-
dertaking (JU) under Grant Agreement No 101118139. The JU receives support from the European Union’s Horizon Europe Programme.

