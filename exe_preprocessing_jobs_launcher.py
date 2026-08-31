"""
Submits all of the tide_correction_model_generation rheological combinations as a Slurm array.
All Slurm settings are intentionally hard-coded here.
"""

from pathlib import Path
from shlex import quote
from subprocess import CalledProcessError, run

from gins_partials import quote_slurm_arg

# ---------------------------------------------------------------------------
# Hard-coded job configuration.
# ---------------------------------------------------------------------------

DEFAULT_CLUSTER_PYTHON_MODULE = "python/3.11.10"
DEFAULT_CLUSTER_VENV = Path(".").parent.joinpath("alna_venv")
JOB_NAME = "generate_tide_model"
SLURM_FILE = Path("run_plot_gravity.sbatch")
WALLTIME = "00:30:00"
MEM = "16G"
CPUS_PER_TASK = 1
ACCOUNT = "grgs"
MODELS_PATH = (
    "/home/users/qt/rousselm/repositories/common_data/tests/solid_earth_numerical_model/for_gins"
)
TIDE_CORRECTION_MODEL_SCRIPT = Path("tide_correction_model_generation.py").resolve()
LAUNCHER_PATH = Path(__file__).resolve()


def shell_join_multiline(cmd: list[str]) -> str:
    """
    Renders a shell command over multiple lines.
    """

    return " \\\n    ".join(quote_slurm_arg(x) for x in cmd)


def make_slurm_script(workdir: Path = Path(".")) -> Path:
    """
    Generates the Slurm script used by all array tasks.
    """

    slurm_file = SLURM_FILE.resolve()
    slurm_file.parent.mkdir(parents=True, exist_ok=True)
    cluster_python = Path(DEFAULT_CLUSTER_VENV).resolve() / "bin" / "python"
    logs_dir = workdir.joinpath("logs").resolve()
    preamble = f"""#!/bin/bash

#SBATCH --job-name={JOB_NAME}
#SBATCH --time={WALLTIME}
#SBATCH --mem={MEM}
#SBATCH --cpus-per-task={CPUS_PER_TASK}
#SBATCH --output={logs_dir}/slurm_%A_%a.out
#SBATCH --error={logs_dir}/slurm_%A_%a.err

set -euo pipefail

echo "Job started on: $(hostname)"
echo "SLURM_JOB_ID=${{SLURM_JOB_ID:-unset}}"
echo "SLURM_ARRAY_TASK_ID=${{SLURM_ARRAY_TASK_ID:-unset}}"

if [ -z "${{SLURM_ARRAY_TASK_ID:-}}" ]; then
    echo "This script must be submitted as a Slurm job array."
    echo "Expected SLURM_ARRAY_TASK_ID to be set."
    exit 1
fi

mkdir -p {quote(str(logs_dir))}
cd {quote(str(workdir.resolve()))}

if command -v module >/dev/null 2>&1; then
    module purge
    module load {quote(DEFAULT_CLUSTER_PYTHON_MODULE)}
fi

source {quote(str(Path(DEFAULT_CLUSTER_VENV) / "bin" / "activate"))}

"""

    task_commands = []
    task_id = 0

    for file_path in Path(MODELS_PATH).glob("*"):

        if "periods" in file_path.name:

            continue

        task_id += 1
        command = [
            str(cluster_python),
            str(TIDE_CORRECTION_MODEL_SCRIPT),
            "--file_path",
            str(file_path),
        ]

        task_commands.append(f"""if [ "$SLURM_ARRAY_TASK_ID" -eq {task_id} ]; then
    {shell_join_multiline(command)}
    echo "Job finished."
    exit 0
fi""")

    script = preamble + "\n".join(task_commands) + """
echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
exit 1
"""

    slurm_file.write_text(script, encoding="utf-8")
    return slurm_file


def submit_slurm(workdir: Path = Path("."), n_jobs_max: int = 500) -> None:
    """Generate and submit the eight-task Slurm array."""

    logs_dir = workdir.joinpath("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    Path(MODELS_PATH).parent.parent.parent.joinpath("tide_models")
    slurm_file = make_slurm_script(workdir=workdir)
    n_jobs = len(Path(MODELS_PATH).glob("*")) - 1
    array_spec = f"1-{n_jobs}%{n_jobs_max}"

    cmd = [
        "sbatch",
        f"--account={ACCOUNT}",
        f"--array={array_spec}",
        str(slurm_file),
    ]

    print("Submitting Slurm job array:")
    print(" ".join(quote(str(x)) for x in cmd))

    try:

        result = run(cmd, text=True, capture_output=True, check=True)

    except CalledProcessError as exc:

        print("sbatch failed.")
        print(f"Generated Slurm script: {slurm_file}")

        if exc.stdout:

            print("sbatch stdout:")
            print(exc.stdout.rstrip())

        if exc.stderr:

            print("sbatch stderr:")
            print(exc.stderr.rstrip())

        raise

    print("sbatch output:")
    print(result.stdout.strip())
    print(f"Generated Slurm script: {slurm_file}")


if __name__ == "__main__":

    submit_slurm()
