"""
Experiment runner for tiered task validation.

Designed for use in Colab or local execution of system_merged_execution_v3_6.
Usage:
    python run_tier_experiments.py --tier 1
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[2]
COLAB_ROOT = Path("/content/computer-vision")


def ensure_repo_on_syspath() -> None:
    """Make sure repository roots are importable."""
    for candidate in (REPO_ROOT, COLAB_ROOT):
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.append(candidate_str)


def load_execution_module() -> ModuleType:
    """Load execution API from either a Python module or the notebook."""
    ensure_repo_on_syspath()

    try:
        return importlib.import_module("system_merged_execution_v3_6")
    except ModuleNotFoundError:
        nb_path_candidates = [
            COLAB_ROOT / "system_merged_execution_v3_6.ipynb",
            REPO_ROOT / "system_merged_execution_v3_6.ipynb",
        ]
        nb_path = next((p for p in nb_path_candidates if p.exists()), None)
        if nb_path is None:
            raise

        try:
            import nbformat  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "nbformat is required to import the execution notebook. "
                "Install it with `pip install nbformat`."
            ) from exc

        notebook = nbformat.read(nb_path, as_version=4)
        module = ModuleType("system_merged_execution_v3_6_notebook")
        env: dict = {}
        for cell in notebook.cells:
            if cell.get("cell_type") == "code":
                exec(cell.get("source", ""), env)  # noqa: S102

        for attr in (
            "prepare_execution_cycle",
            "run_execution_cycle",
            "export_task_contexts",
            "tracker",
        ):
            if attr not in env:
                raise AttributeError(
                    f"Notebook at {nb_path} does not define `{attr}`; "
                    "run the notebook once to materialize the API."
                )
            setattr(module, attr, env[attr])
        return module


def load_tier_tasks(tier: int) -> List[str]:
    """Load task IDs defined for a tier."""
    tier_file = REPO_ROOT / f"multi-agent/experiments/tier{tier}_tasks.json"
    with open(tier_file, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return [task["task_id"] for task in data.get("tasks", [])]


def run_tier(tier: int) -> None:
    """Run the execution cycle for the specified tier."""
    module = load_execution_module()
    required_attrs = (
        "prepare_execution_cycle",
        "run_execution_cycle",
        "export_task_contexts",
        "tracker",
    )
    missing = [attr for attr in required_attrs if not hasattr(module, attr)]
    if missing:
        raise AttributeError(
            f"Execution module missing attributes: {', '.join(missing)}"
        )

    prepare_execution_cycle = getattr(module, "prepare_execution_cycle")
    run_execution_cycle = getattr(module, "run_execution_cycle")
    export_task_contexts = getattr(module, "export_task_contexts")
    tracker = getattr(module, "tracker")

    task_ids = load_tier_tasks(tier)
    if not task_ids:
        raise ValueError(f"No tasks found for tier {tier}.")

    banner = f"{'=' * 60}\nRUNNING TIER {tier} EXPERIMENTS\n{'=' * 60}"
    print(banner)
    print(f"Tasks to run: {task_ids}\n")

    ids = prepare_execution_cycle(selected_task_ids=task_ids)
    run_execution_cycle(ids)

    export_path = export_task_contexts(f"tier{tier}_results")
    print(f"\nResults exported to: {export_path}\n")

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    tracker.get_summary()
    tracker.get_retry_stats()


def parse_args(args: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tier",
        type=int,
        required=True,
        choices=[1, 2, 3, 4],
        help="Tier number to execute.",
    )
    return parser.parse_args(args)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    run_tier(args.tier)


if __name__ == "__main__":  # pragma: no cover
    main()
