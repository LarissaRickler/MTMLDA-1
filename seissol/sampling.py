import multiprocessing
import os
import pickle
from functools import partial

import numpy as np
import umbridge as ub

import models.posterior_pto_wrapper as wrapper
import src.mtmlda.mcmc as mcmc
import src.mtmlda.sampler as sampler


# ==================================================================================================
def execute_mtmlda_run(
    process_id,
    run_settings,
    proposal_settings,
    accept_rate_settings,
    sampler_setup_settings,
    sampler_run_settings,
    model_settings,
    prior_settings,
    likelihood_settings,
):
    modify_process_dependent_settings(
        process_id, proposal_settings, sampler_setup_settings, sampler_run_settings, prior_settings
    )
    models, prior = set_up_models(model_settings, prior_settings, likelihood_settings)
    sampler_run_settings.initial_state = prior.sample()

    mtmlda_sampler = set_up_sampler(
        process_id,
        run_settings,
        proposal_settings,
        accept_rate_settings,
        sampler_setup_settings,
        models,
    )
    mcmc_chain = mtmlda_sampler.run(sampler_run_settings)
    save_results(process_id, run_settings, mcmc_chain, mtmlda_sampler)


def modify_process_dependent_settings(
    process_id, proposal_settings, sampler_setup_settings, sampler_run_settings, prior_settings
) -> None:
    proposal_settings.rng_seed = process_id
    sampler_setup_settings.rng_seed_mltree = process_id
    sampler_run_settings.rng_seed_node_init = process_id
    prior_settings.rng_seed = process_id

    if process_id != 0:
        sampler_setup_settings.logfile_path = None
        sampler_setup_settings.mltree_path = None
        sampler_setup_settings.do_printing = False


def set_up_models(model_settings, prior_settings, likelihood_settings):
    configs = model_settings.configs
    pto_model = ub.HTTPModel(model_settings.address, model_settings.name)
    prior = wrapper.UninformLogPrior(prior_settings.parameter_intervals, prior_settings.rng_seed)
    likelihood = wrapper.GaussianLogLikelihood(
        pto_model, likelihood_settings.data, likelihood_settings.covariance
    )
    model_wrapper = wrapper.LogPosterior(prior, likelihood)
    models = [partial(model_wrapper, config=mesh_file_name) for mesh_file_name in configs]
    
    return models, prior


def set_up_sampler(
    process_id,
    run_settings,
    proposal_settings,
    accept_rate_settings,
    sampler_setup_settings,
    models,
):
    ground_proposal = mcmc.RandomWalkProposal(
        proposal_settings.step_width,
        proposal_settings.covariance,
        proposal_settings.rng_seed,
    )
    accept_rate_estimator = mcmc.MLAcceptRateEstimator(
        accept_rate_settings.initial_guess,
        accept_rate_settings.update_parameter,
    )
    mtmlda_sampler = sampler.MTMLDASampler(
        sampler_setup_settings,
        models,
        accept_rate_estimator,
        ground_proposal,
    )
    if run_settings.rng_state_load_file_stem is not None:
        rng_state_file = (
            run_settings.result_directory_path
            / run_settings.rng_state_load_file_stem.with_name(
                f"{run_settings.rng_state_load_file_stem.name}_{process_id}.pkl"
            )
        )
        with rng_state_file.open("rb") as rng_state_file:
            rng_states = pickle.load(rng_state_file)
        mtmlda_sampler.set_rngs(rng_states)
    return mtmlda_sampler


def save_results(process_id, run_settings, mcmc_trace, mtmlda_sampler) -> None:
    os.makedirs(run_settings.result_directory_path, exist_ok=run_settings.overwrite_results)
    chain_file = run_settings.result_directory_path / run_settings.chain_file_stem.with_name(
        f"{run_settings.chain_file_stem.name}_{process_id}.npy"
    )
    np.save(chain_file, mcmc_trace)

    if run_settings.rng_state_save_file_stem is not None:
        rng_state_file = (
            run_settings.result_directory_path
            / run_settings.rng_state_save_file_stem.with_name(
                f"{run_settings.rng_state_save_file_stem.name}_{process_id}.pkl"
            )
        )
        with rng_state_file.open("wb") as rng_state_file:
            pickle.dump(mtmlda_sampler.get_rngs(), rng_state_file)


# --------------------------------------------------------------------------------------------------
def run(
    run_settings,
    proposal_settings,
    accept_rate_settings,
    sampler_setup_settings,
    sampler_run_settings,
    model_settings,
    prior_settings,
    likelihood_settings,
) -> None:
    process_ids = range(run_settings.num_chains)

    execute_mtmlda_on_procs = partial(
        execute_mtmlda_run,
        run_settings=run_settings,
        proposal_settings=proposal_settings,
        accept_rate_settings=accept_rate_settings,
        sampler_setup_settings=sampler_setup_settings,
        sampler_run_settings=sampler_run_settings,
        model_settings=model_settings,
        prior_settings=prior_settings,
        likelihood_settings=likelihood_settings,
    )

    with multiprocessing.Pool(processes=run_settings.num_chains) as process_pool:
        process_pool.map(execute_mtmlda_on_procs, process_ids)
