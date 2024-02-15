from dummy_seissol import dummy_settings, sampling


def main():
    sampling.run(
        dummy_settings.run_settings,
        dummy_settings.proposal_settings,
        dummy_settings.accept_rate_settings,
        dummy_settings.sampler_setup_settings,
        dummy_settings.sampler_run_settings,
        dummy_settings.model_settings,
        dummy_settings.prior_settings,
        dummy_settings.likelihood_settings,
    )

if __name__ == "__main__":
    main()



