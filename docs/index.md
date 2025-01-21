# MTMLDA [<img src="images/uq_logo.png" width="200" height="100" alt="UQ at KIT" align="right">](https://www.scc.kit.edu/forschung/uq.php)

MTMLDA is a specialized implementation of the *Multilevel Delayed Acceptance* (MLDA) algorithm. MLDA is a multilevel *Markov Chain Monte Carlo* method, which has been proposed [here](https://doi.org/10.1137/22M1476770). Like other multilevel sampling procedures, MLDA utilizes a hierarchy of models that approximate the target distribution with varying fidelities.

The implemented version comprises within-chain parallelism through [prefetching](https://www.tandfonline.com/doi/abs/10.1198/106186006X100579), the expansion of possible future states of the Markov chain in a binary decision tree. The target density evaluations at these states can be performed in parallel, potentially increasing MCMC execution speed. This is particularly useful for scenarios where burn-in is significant, such that parallelizaion through multiple chains is inefficient. We combine MLDA with asynchronous prefetching, to make full use of a hierarchy of models with potentially vastly different evaluation times. The theoretical background and conceptual setup of the implementation can be found in the accompanying publication:

 > ***Scalable Bayesian Inference of Large Simulations via Asynchronous Prefetching Multilevel Delayed Acceptance (to be published)***.

!!! warning
    MTMLDA is a library developed in the course of a research project, not as a dedicated tool. As
    such such, it has been tested for a number of example use cases, but not with an exhaustive test suite. Therefore, we currently do not intend to upload this library to a public index.

## Installation and Development

The library in this repository is a Python package readily installable via `pip`, simply run
```bash
python pip install .
```
For development, we recommend using the great [uv](https://docs.astral.sh/uv/) project management tool, for which MTMLDA provides a universal lock file. To set up a reproducible environment, run 
```bash
uv sync --all-groups
```

To render images from generated dot files, you also need to have [Graphviz](https://graphviz.org/) installed on your system.

## Documentation

#### Usage

Under Usage, we provide a [tutorial](usage/tutorial.md) on how to use MTMLDA.

#### API Reference

The API reference contains detailed explanations of all software components of MTMLDA, and how to use them.

#### Examples

We provide [runnable examples](https://github.com/UQatKIT/mtmlda/tree/main/examples) in our Github repository.

## License

MTMLDA is distributed under the [MIT](https://choosealicense.com/licenses/mit/) license.

It has been developed as part of the innovation study **ScalaMIDA**. It has received funding through the **Inno4scale** project, which is funded by the European High-Performance Computing Joint Un-
dertaking (JU) under Grant Agreement No 101118139. The JU receives support from the European Union’s Horizon Europe Programme.