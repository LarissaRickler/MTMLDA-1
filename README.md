# MTMLDA: Within-Chain Parallelism for Multilevel MCMC based on Prefetching

This repository contains a  specialized implementation of the *Multilevel Delayed Acceptance* (MLDA) algorithm. MLDA is a multilevel *Markov Chain Monte Carlo* algorithm, which has been proposed [here](https://doi.org/10.1137/22M1476770). Like other multilevel sampling procedures, MLDA utilizes a hierarchy of models that approximate the target distribution with varying fidelities. The implemented version comprises within-chain parallelism through [prefetching](https://www.tandfonline.com/doi/abs/10.1198/106186006X100579), the expansion of possible future states of the Markov chain in a binary decision tree. The target density evaluations at these states can be performed in parallel, potentially increasing MCMC execution. This is particularly useful for scenarios where burn-in is significant, such that parallelizaion through multiple chains is inefficient. We combine MLDA with asynchronous prefetching, to make full use of a hierarchy of models with potentially vastly different evaluation times. The theoretical background and conceptual setup of the implementation can be found in the accompanying publication, "Scalable Bayesian Inference of Large Simulations via Asynchronous Prefetching Multilevel Delayed Acceptance".

## Installation

The library in this repository is a Python package readily installable via `pip`, simply run
```python
pip install .
```
in the root directory. Alternatively, you may use [UV](https://docs.astral.sh/uv/), which has been used for the setup of the project.

## Usage

`mtmlda` is a modular, low-level library, whose functionality is implemented in the `core` submodule. Its main component is the `MTMLDASampler`sampler object, which requires, next to MCMC-specific components, a list of callables resembling the model hierarchy. To accommodate different scenarios, we provide the user with the possibility to implement an object with the interface of `ApplicationBuilder` in the `components` submodule. In addition, the `components` module provides interfaces and exemplary implementations of components that are typically required to set up a hierarchy of Bayesian posterior densities. The builder can then be used, in combination with a `settings.py` file, to set up a `ParallelRunner` object in the `run`. This object sets up the model hierarchy and sampler. Subsequently, it can run multiple MLDA chains in parallel, each of which is potentially parallelized though prefetching. All parallelization is effectively handled through the asynchronous calls to the hierarchy of models. To process these request effectively, we rely on the [UM-Bridge](https://um-bridge-benchmarks.readthedocs.io/en/docs/) framework, which can dispatch the calls to any sort of external framework, up to HPC architectures. After results of an mtmlda run comprise detailed logfiles (run and debug), as well as the generated chains, and potentially the Markov trees generated during the sampling. Data and trees can be analyzed by the `PostProcessor` in the `run`module.

All mentioned components have extensivee in-code documentation, which we refer the reader to for moder detailed information on their usage and implementation. In the following, we focus on the execution of the example application `example_01`. Any application to be processed by the standard workflow has to comprise a `settings.py`and a `builder.py` file. The settings file is the unified entry point for the configuration of the models, the builder and the sampler. It implements data classes that serve as input settings for the respective components.
For the MCMC client to work, we need to start a server-side model hierarchy to be called. This is implemented as an UMBridge server in `simulation_model.py`, execute in a separate terminal session with
```
python simulation_model.py
```

To then execute a sampling application, simply invoke the `run.py` script with the application direction as argument,
```
python run.py -app examples/example_01
```
The run script sets up a parallel runner for the execution and conducts the MLDA run. Subsequently, you can analysize and visualize results with 
```
python postprocessing.py -app examples_example_01
```

Lastly, we discuss in more details the actual settings specified in `settings.py`, with the example application `example_01`.

## License

This Software is distributed under the [MIT](https://choosealicense.com/licenses/mit/) license.

It has been developed as part of the innovation study **ScalaMIDA**. It has received funding through the **Inno4scale** project, which is funded by the European High-Performance Computing Joint Un-
dertaking (JU) under Grant Agreement No 101118139. The JU receives support from the European Union’s Horizon Europe Programme.

