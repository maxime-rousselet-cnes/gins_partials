"""Submit the eight plot_gravity_test boolean combinations as a Slurm array.

Usage:
    python exe_plot_jobs_launcher.py

The eight array tasks cover every possible combination of:
    --alpha
    --delta
    --tau_m

All Slurm settings are intentionally hard-coded here.
"""

from pathlib import Path
from shlex import quote
from subprocess import CalledProcessError, run

# ---------------------------------------------------------------------------
# Hard-coded job configuration.
# ---------------------------------------------------------------------------

DEFAULT_CLUSTER_PYTHON_MODULE = "python/3.11"
JOB_NAME = "plot_gravity"
SLURM_FILE = Path("run_plot_gravity.sbatch")
WALLTIME = "24:00:00"
MEM = "4G"
CPUS_PER_TASK = 1
ACCOUNT = "grgs"
ROOT = "/work/GRGS/users/rousselm/dynamo/rheology/solution"

PLOT_SCRIPT = Path("plot_gravity_test.py").resolve()
LAUNCHER_PATH = Path(__file__).resolve()
N_TASKS = 8


def combinations() -> list[tuple[bool, bool, bool]]:
    """Return all alpha/delta/tau_m combinations in task-id order."""

    return [
        (alpha, delta, tau_m)
        for alpha in (False, True)
        for delta in (False, True)
        for tau_m in (False, True)
    ]


def quote_slurm_arg(value: str) -> str:
    """Quote a command argument, preserving the Slurm array-variable syntax."""

    if value == "${SLURM_ARRAY_TASK_ID}":
        return '"${SLURM_ARRAY_TASK_ID}"'

    return quote(str(value))


def shell_join_multiline(cmd: list[str]) -> str:
    """Render a shell command over multiple lines."""

    return " \\\n    ".join(quote_slurm_arg(x) for x in cmd)


def make_slurm_script(workdir: Path = Path(".")) -> Path:
    """Generate the Slurm script used by all eight array tasks."""

    slurm_file = SLURM_FILE.resolve()
    slurm_file.parent.mkdir(parents=True, exist_ok=True)

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

"""

    task_commands = []
    for task_id, (alpha, delta, tau_m) in enumerate(combinations(), start=1):
        flags = []
        if alpha:
            flags.append("--alpha")
        if delta:
            flags.append("--delta")
        if tau_m:
            flags.append("--tau_m")

        command = [
            "python",
            str(PLOT_SCRIPT),
            "--root",
            ROOT,
            *flags,
        ]

        task_commands.append(f"""if [ "$SLURM_ARRAY_TASK_ID" -eq {task_id} ]; then
    echo "Task {task_id}: alpha={str(alpha).lower()} delta={str(delta).lower()} tau_m={str(tau_m).lower()}"
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


def submit_slurm(workdir: Path = Path(".")) -> None:
    """Generate and submit the eight-task Slurm array."""

    logs_dir = workdir.joinpath("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    slurm_file = make_slurm_script(workdir=workdir)

    array_spec = f"1-{N_TASKS}"

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
