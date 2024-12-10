# MTMLDA: Within-Chain Parallelism for Multilevel MCMC based on Prefetching

This repository contains a  specialized implementation of the *Multilevel Delayed Acceptance* (MLDA) algorithm. MLDA is a multilevel *Markov Chain Monte Carlo* algorithm, which has been proposed [here](https://doi.org/10.1137/22M1476770). Like other multilevel sampling procedures, MLDA utilizes a hierarchy of models that approximate the target distribution with varying fidelities. The implemented version comprises within-chain parallelism through [prefetching](https://www.tandfonline.com/doi/abs/10.1198/106186006X100579), the expansion of possible future states of the Markov chain in a binary decision tree. The target density evaluations at these states can be performed in parallel, potentially increasing MCMC execution. This is particularly useful for scenarios where burn-in is significant, such that parallelizaion through multiple chains is inefficient. We combine MLDA with asynchronous prefetching, to make full use of a hierarchy of models with potentially vastly different evaluation times. The theoretical background and conceptual setup of the implementation can be found in the accompanying publication, "Scalable Bayesian Inference of Large Simulations via Asynchronous Prefetching Multilevel Delayed Acceptance".

## Installation

The library in this repository is a Python package readily installable via `pip`, simply run
```python
pip install .
```
in the root directory. Alternatively, you may use [UV](https://docs.astral.sh/uv/), which has been used for the setup of the project.

## Usage

`mtmlda` is a modular, low-level library, whose functionality is implemented in the `core` submodule. Its main component is the `MTMLDASampler`sampler object, which requires, next to MCMC-specific components, a list of callables resembling the model hierarchy (**log densities**). To accommodate different scenarios, we provide the user with the possibility to implement an object with the interface of `ApplicationBuilder` in the `components` submodule. In addition, the `components` module provides interfaces and exemplary implementations of components that are typically required to set up a hierarchy of Bayesian posterior densities. The builder can then be used, in combination with a `settings.py` file, to set up a `ParallelRunner` object in the `run`. This object sets up the model hierarchy and sampler. Subsequently, it can run multiple MLDA chains in parallel, each of which is potentially parallelized though prefetching. All parallelization is effectively handled through the asynchronous calls to the hierarchy of models. To process these request effectively, we rely on the [UM-Bridge](https://um-bridge-benchmarks.readthedocs.io/en/docs/) framework, which can dispatch the calls to any sort of external framework, up to HPC architectures. After results of an mtmlda run comprise detailed logfiles (run and debug), as well as the generated chains, and potentially the Markov trees generated during the sampling. Data and trees can be analyzed by the `PostProcessor` in the `run`module.

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

Lastly, we discuss in more details the actual settings specified in `settings.py`, for the example application `example_01`.

**The first block of settings is identical for all applications, it comprises the settings for the MTMLDA algorithm.**

**`ParallelRunSettings`** configures the parallel runner wrapper:
- `num_chains` determines the number of parallel MLDA chains to sample
- `chain_save_path` indicates the directory where to store the resulting samples in `npy` file format
- `chain_load_path` can point to existing samples for re-initialization of a sampling run
- `rng_state_save_path` can be set to store the numpy RNGs used during sampling in `pkl` format
- `rng_state_load_path` can fetch those RNGs for re-initialization
- `overwrite_chains` indicates whether overwriting existing chain data is permitted
- `overwrite_rng_states`indicates the same for the RNGs

**`SamplerSetupSettings`** configures the initialization of the MLDA sampler:
- `num_levels` indicates the depth of the multilevel model hierarchy
- `subsampling_rates` determines the length of subchains on respective levels, from coarse to fine, has to be of length `num_levels`
- `max_tree_height` is a technical setting, restricting the Markov tree in the algorithm to a maximum depth. This depth should usually not be reached.
- `underflow_threshold` is the value for the obtained log densities below which the corresponding density is treated as zero. The threshold is implemented for numerical stability.
- `rng_seed_mltree` is the seed of the RNG that is used for the uniform numbers for comparison in accept/reject steps. Every new node in  a Markov tree is equipped with such a random number.
- `rng_seed_node_init` is the seed of the RNG used for initialization of the first node in the Markov tree. The RNG samples an initial state, only necessary if such a state is not provided (see `SamplerRunSettings`)
- `mltree_path`indicates where to store exported Markov trees, if wanted

**`SamplerRunSettings`** configures the MLDA run for an initialized sampler:
- `num_samples` denotes the number of fine-level samples to generate
- `initial_state` defines the initial parameter value in the chain
- `num_threads` is the number of parallel workers for prefetching
- `print_interval` determines the stride on samples after which info is sent to the logger

**`LoggerSettings`** configures the MTMLDA logger:
- `do_printing` determines if the run logger info is printed to console
- `logfile_path` indicates where run logger info is stored, if wanted 
- `debugfile_path`indicates where debug logger info is stored, if wanted

**The second block of settings goes into the application builder, it is therefore application-specific.**

For the current example, we assume that an Umbridge server provides us with a 4D -> 4D *parameter-to-observable map*. From this the builder constructs a Gaussian likelihood, and forms a posterior through the combination with a uniform prior.

**`InverseProblemSettings`** configures the initialization of the model hierarchy for MLDA:
- `prior_intervals` gives dimension-wise intervals for the uniform prior
- `prior_rng_seed` is the seed for the prior-internal RNG, which is used to draw samples from it
- `likelihood_data` is the data vector used to construct a misfit with the output of the PTO map
- `likelihood_covariance` defines the covariance matrix used to generate a Gaussian likelihood from the misfit vector
- `ub_model_configs` defines the values of the `config` argument of calls to the UMBridge model server for each model in the hierarchy.
- `ub_model_address` is the address of the UMBridge server
- `ub_model_name` is the name of the UMBridge server

**`SamplerComponentSettings`** defines the configuration of the components to be passed to the MLDA sampler, these are the proposal and the accept rate estimator. In this example, 
we utilize a Metropolis `RandomWalkProposal` and a `StaticAcceptRateEstimator`. We refer to the documentation of these components for further details.
- `proposal_step_width` is the step width of the random walk proposal
- `proposal_covariance` is the covariance matrix for the Gaussian proposal step
- `proposal_rng_seed` is the seed initializing the RNG for proposal step sampling
- `accept_rates_initial_gues` sets the initial accept rate estimates for each level in the model hierarchy
- `accept_rates_update_parameter` is a factor for the decrease/increase of the accept rate estimate of a level if a proposal is accepted or rejected through an MCMC decision
  
**`InitialStateSettings`** determines how to initialize the Markov Chain for MLDA. For our application, it is empty, as the initial states are simply sampled from the prior (if not provided explicitly)


**The last block of settings is again generic, it comprises the configuration of the postprocessor.**

**`PostprocessorSettings`**:
- `chain_directory` points to the sampled chains to postprocess
- `tree_directory`points to exported Markov trees for rendering, if wanted
- `output_data_directory` says where to store postprcessed data
- `visualization_directory` says where to store visualizations
- `acf_lag_max` determines up to which lag autocorrelation functions should be computed

**Note**: Several settings may be set and or modified by the parallel run wrapper, depending on the index of the chain they are used for.


## License

This Software is distributed under the [MIT](https://choosealicense.com/licenses/mit/) license.

It has been developed as part of the innovation study **ScalaMIDA**. It has received funding through the **Inno4scale** project, which is funded by the European High-Performance Computing Joint Un-
dertaking (JU) under Grant Agreement No 101118139. The JU receives support from the European Union’s Horizon Europe Programme.

