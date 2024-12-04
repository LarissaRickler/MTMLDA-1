import os
import shlex
import subprocess
import time


# ===================================================================================================
def start_server(command: str) -> subprocess.Popen:
    print("1) Start simulation model server:", end=" ", flush=True)
    simulation_server_proc = subprocess.Popen(shlex.split(command), stdout=subprocess.DEVNULL)
    time.sleep(1)
    print("Done", flush=True)
    return simulation_server_proc


# ---------------------------------------------------------------------------------------------------
def run_command(command: str, message: str) -> None:
    print(message, end=" ", flush=True)
    proc = subprocess.run(shlex.split(command), capture_output=True)
    try:
        proc.check_returncode()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(proc.stderr.decode()) from exc
    else:
        print("Done", flush=True)


# ---------------------------------------------------------------------------------------------------
def run_application(path: str) -> None:
    print(f"===== Running application {path} =====", flush=True)
    simulation_server_command = f"python {path}/simulation_model.py"
    sampling_run_command = f"python run.py -app {path}"
    postprocessing_command = f"python postprocessing.py -app {path}"

    try:
        simulation_server_proc = start_server(simulation_server_command)
        run_command(sampling_run_command, "2) Start MTMLDA Sampling:")
        run_command(postprocessing_command, "3) Start Postprocessing:")
    finally:
        print("4) Stop simulation model server:", end=" ", flush=True)
        simulation_server_proc.kill()
        print("Done", flush=True)
        print("", flush=True)


# ===================================================================================================
def main() -> None:
    applications = ["example_01", "example_02", "example_03"]

    for application_path in applications:
        run_application(application_path)


if __name__ == "__main__":
    main()
