from seissol import settings, sampling


def main():
    sampling.run(
        settings.run_settings,
        settings.proposal_settings,
        settings.accept_rate_settings,
        settings.sampler_setup_settings,
        settings.sampler_run_settings,
        settings.model_settings,
        settings.prior_settings,
        settings.likelihood_settings,
    )

if __name__ == "__main__":
    main()



