"""Utility functions for the benchmark comparison study."""

import os
import shutil


def time_saving(experiment_start_time, experiment_end_time, save_dir: str):
    """Save the time it took to run an experiment."""
    elapsed = experiment_end_time - experiment_start_time
    elapsed_minutes = elapsed / 60.0
    elapsed_hours = elapsed / 3600.0

    print(
        f"Time taken: {elapsed:.2f} seconds ({elapsed_minutes:.2f} min, {elapsed_hours:.2f} hr)\n"
    )

    # Save timing to text file
    timing_path = os.path.join(save_dir, "timing.txt")
    with open(timing_path, "w") as f:
        f.write(f"Time taken (seconds): {elapsed:.2f}\n")
        f.write(f"Time taken (minutes): {elapsed_minutes:.2f}\n")
        f.write(f"Time taken (hours): {elapsed_hours:.2f}\n")


def copy_data_files_to_results_dir(
    data_dir: str, target_dir: str, filenames: list[str] | None = None
):
    """Copy data files from a data directory to a target directory."""
    if not filenames:
        filenames = [
            "noiseless_data.csv",
            "all_data_X.npy",
            "all_initial_conditions.npy",
            "data_plot.npy",
        ]
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file in filenames:
                # Create corresponding directory in target_dir
                relative_path = os.path.relpath(root, data_dir)
                new_dir = os.path.join(target_dir, relative_path)
                os.makedirs(new_dir, exist_ok=True)

                # Copy the file
                src_file = os.path.join(root, file)
                dst_file = os.path.join(new_dir, file)
                shutil.copy2(src_file, dst_file)
