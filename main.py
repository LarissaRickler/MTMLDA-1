from application import sampling
from seissol import settings_seissol as settings


def main():
    sampling.run(
        settings.run_settings,
        settings.proposal_settings,
        settings.accept_rate_settings,
        settings.sampler_setup_settings,
        settings.sampler_run_settings,
        settings.models
    )

if __name__ == "__main__":
    main()



