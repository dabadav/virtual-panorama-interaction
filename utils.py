# Load libraries
import fnmatch  # Pattern matching
import json  # Load JSON filetype
import os
import re  # Regular expressions

import matplotlib.pyplot as plt
import yaml  # Load Yaml filetype

##### Loading


def load_yaml(file_path):
    with open(file_path) as f:
        try:
            mapping = yaml.safe_load(f)
            return mapping
        except yaml.YAMLError as exc:
            print(exc)
            return None


def load_json(file_path):
    """Safe load json file"""
    with open(file_path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file: {file_path}")
            print(e)
            return None


def match_files(data_dir, pattern):
    """Given directory find matching files"""
    files = [
        os.path.join(data_dir, filename)
        for filename in os.listdir(data_dir)
        if fnmatch.fnmatch(filename, pattern)
    ]
    return files


def load_json_files(data_dir, pattern):
    """Loads JSON files from a directory matching a filename pattern.

    Args:
        data_dir: Path to the directory containing JSON files.
        pattern: Filename pattern to match (e.g., 'data_*.json').

    Returns:
        A list of dictionaries, where each dictionary represents a matched JSON file.
    """

    # List all files in folder
    json_files = match_files(data_dir, pattern)

    # List of python objects
    data = [load_json(file_path) for file_path in json_files if file_path]

    return data


def extract_id(filename):
    """Extracts the numeric ID from the filename."""
    match = re.search(r"_(\d+)\.json$", filename)
    return match.group(1) if match else None


def extract_id_files(data_dir, pattern="Log_Survey_BB_*.json"):
    """
    Extracts numeric IDs from filenames matching a given pattern.

    Args:
        data_dir (str): Path to the directory containing files.
        pattern (str): Filename pattern to match (default: 'Log_Survey_BB_*.json').

    Returns:
        list: A list of extracted numeric IDs as strings.
    """

    # List all matching files
    files = match_files(data_dir, pattern)

    # Extract IDs and filter out None values
    ids = [extract_id(file) for file in files]

    return ids


##### Interaction Utils


def sample(df, visitor_id):
    return df[df["visitor_id"] == visitor_id]


def time_unit(unit: str):
    match unit:
        case "s":
            dividend = 1
        case "min":
            dividend = 60
        case "h":
            dividend = 60 * 60
        case _:
            raise ValueError(f"Unsupported time unit: {unit}")

    return dividend


def plot_sample(sample, unit="min"):

    dividend = time_unit(unit)
    times = (sample["time"] - sample["time"].min()) // dividend

    # Plot all actions at y = 0
    plt.figure(figsize=(12, 2))
    plt.scatter(times, [0] * len(times), marker="|", s=200)

    # Beautify
    plt.yticks([])  # Remove y-axis ticks
    plt.xlabel(f"Time ({unit})")
    plt.title("Global Action Raster (All Actions)")
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def remove_inactivity(group):
    if group["inactive"].any():
        # Index of the first inactivity
        first_inactive_idx = group.index[group["inactive"]].min()
        # Keep only rows before that
        return group.loc[: first_inactive_idx - 1]
    return group  # keep all if no inactivity
