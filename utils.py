# Load libraries
from typing import Literal
import fnmatch  # Pattern matching
import json  # Load JSON filetype
import os
import re  # Regular expressions

import matplotlib.pyplot as plt
import yaml  # Load Yaml filetype
import pandas as pd
from pathlib import Path

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

def plot_sample_by_category(
    sample: pd.DataFrame,
    unit="min",
    save_path: str | None = None,
    all_categories: list[str] | None = ["Content", "Navigation", "Survey", "Settings", "Touch"],
    category_colors: dict[str, str] | None = {
        "Content": "blue",
        "Navigation": "orange",
        "Survey": "green",
        "Settings": "purple",
        "Touch": "gray"
    },
):

    # Normalize time
    dividend = time_unit(unit)
    base_time = sample["time"].min()
    sample = sample.copy()
    sample["time_unit"] = (sample["time"] - base_time) / dividend
    inactivity_regions = get_inactivity_regions(sample)

    # Categories as vertical lines
    vline_categories = {"Start": "green", "Finish": "red"}

    # Use consistent y-levels
    non_vline_categories = [cat for cat in all_categories if cat not in vline_categories] if all_categories else \
        sorted(sample[~sample["category"].isin(vline_categories)].category.unique())
    y_levels = {cat: i for i, cat in enumerate(non_vline_categories)}

    # Plot setup
    # Compute fixed-width scale based on real time duration
    total_duration = (sample["time"].max() - sample["time"].min()) / dividend
    width_per_min = 4
    width = max(6, total_duration * width_per_min)

    plt.figure(figsize=(width, 1 + 0.6 * max(1, len(y_levels))))

    # Plot each category (even if it's empty)
    for cat, y in y_levels.items():
        cat_data = sample[sample["category"] == cat]
        color = category_colors[cat] if category_colors and cat in category_colors else None
        plt.scatter(cat_data["time_unit"], [y] * len(cat_data), label=cat, marker="|", s=200, color=color)

    # Plot Start/Finish vertical lines
    for cat, color in vline_categories.items():
        vline_times = sample[sample["category"] == cat]["time_unit"]
        for t in vline_times:
            plt.axvline(t, color=color, linestyle="--", linewidth=1.5, alpha=0.7, label=cat)

    # Fill inactivity regions
    for start, end in inactivity_regions:


        plt.axvspan(start, end, color='gray', alpha=0.2, zorder=0)

    # Clean legend
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(
        unique.values(),
        unique.keys(),
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False
    )

    plt.yticks(list(y_levels.values()), list(y_levels.keys()))
    plt.xlabel(f"Time ({unit})")
    plt.title("Action Raster by Category")
    plt.grid(True, axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()

def plot_sample_uniform(
    sample: pd.DataFrame,
    unit="min",
    save_path: str | None = None,
    all_categories: list[str] | None = ["Content", "Navigation", "Survey", "Settings", "Touch"],
    category_colors: dict[str, str] | None = {
        "Content": "blue",
        "Navigation": "orange",
        "Survey": "green",
        "Settings": "purple",
        "Touch": "gray"
    },
    fixed_total_duration: float | None = None  # <-- New parameter
):
    # Normalize time
    dividend = time_unit(unit)
    base_time = sample["time"].min()
    sample = sample.copy()
    sample["time_unit"] = (sample["time"] - base_time) / dividend
    inactivity_regions = get_inactivity_regions(sample)

    # Define categories
    vline_categories = {"Start": "green", "Finish": "red"}
    non_vline_categories = [cat for cat in all_categories if cat not in vline_categories] if all_categories else \
        sorted(sample[~sample["category"].isin(vline_categories)].category.unique())
    y_levels = {cat: i for i, cat in enumerate(non_vline_categories)}

    # Calculate duration
    sample_duration = (sample["time"].max() - sample["time"].min()) / dividend
    total_duration = fixed_total_duration if fixed_total_duration is not None else sample_duration
    width_per_min = 4  # inches per minute
    width = min(20, max(6, total_duration * width_per_min))

    # Plot
    plt.figure(figsize=(width, 1 + 0.6 * max(1, len(y_levels))))

    for cat, y in y_levels.items():
        cat_data = sample[sample["category"] == cat]
        color = category_colors[cat] if category_colors and cat in category_colors else None
        plt.scatter(cat_data["time_unit"], [y] * len(cat_data), label=cat, marker="|", s=200, color=color)

    for cat, color in vline_categories.items():
        vline_times = sample[sample["category"] == cat]["time_unit"]
        for t in vline_times:
            plt.axvline(t, color=color, linestyle="--", linewidth=1.5, alpha=0.7, label=cat)

    for start, end in inactivity_regions:
        plt.axvspan(start, end, color='gray', alpha=0.2, zorder=0)

    # Legend and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

    plt.yticks(list(y_levels.values()), list(y_levels.keys()))
    plt.xlabel(f"Time ({unit})")
    plt.title("Action Raster by Category")
    plt.grid(True, axis="x", linestyle="--", alpha=0.4)
    # plt.xlim(0, total_duration)  # <-- Fix horizontal axis
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()

from typing import Iterable

def plot_all_samples_consistently(
    samples: Iterable[pd.DataFrame],
    save_paths: Iterable[str],
    unit: str = "min",
    **kwargs
):
    dividend = time_unit(unit)

    global_min = min(sample["time"].min() for sample in samples)
    global_max = max(sample["time"].max() for sample in samples)
    global_duration = (global_max - global_min) / dividend

    for sample, path in zip(samples, save_paths):
        plot_sample_uniform(
            sample=sample,
            unit=unit,
            save_path=path,
            fixed_total_duration=global_duration,
            **kwargs
        )

def get_inactivity_regions(df):
    regions = []
    sorted_df = df.sort_values("time").reset_index(drop=True)
    inactive = sorted_df["inactive"].fillna(False).tolist()
    time_units = sorted_df["time_unit"].tolist()

    in_region = False
    start_idx = None

    for i in range(len(inactive)):
        if inactive[i] and not in_region:
            in_region = True
            start_idx = i
        elif not inactive[i] and in_region:
            in_region = False
            # Get time before inactivity started, if possible
            start = time_units[start_idx - 1] if start_idx > 0 else time_units[start_idx]
            end = time_units[i]  # first active time after inactivity
            regions.append((start, end))

    # Handle case where inactivity continues to end
    if in_region:
        start = time_units[start_idx - 1] if start_idx > 0 else time_units[start_idx]
        end = time_units[-1]
        regions.append((start, end))

    return regions

def remove_inactivity(group):
    if group["inactive"].any():
        # Index of the first inactivity
        first_inactive_idx = group.index[group["inactive"]].min()
        # Keep only rows before that
        return group.loc[: first_inactive_idx - 1]
    return group  # keep all if no inactivity

def remove_data_before_session_start(group):
    try:
        # Get 'time' of first 'Button_close_Instructions'
        min_time = group[group['action'] == 'Button_close_Instructions']['time'].min()
        # Remove rows with 'time' less than or equal to min_time
        result = group[group['time'] >= min_time]
        return result
    
    except Exception as e:
        print(e)

RemovalWhere = Literal['before', 'after']
ActionWhich = Literal['first', 'last']

def remove_before_or_after_action(
    group: pd.DataFrame,
    action: str,
    which: ActionWhich,
    where: RemovalWhere,
    include: bool = True,
    fallback: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    try:
        actions = group[group['action'] == action]

        if debug:
            print(actions)

        if actions.empty:
            print(f"No action '{action}' found in group '{group['visitor_id'].unique()}'.")
            if fallback:
                return group
            else:
                return None

        # Determine reference time
        ref_time = actions['time'].min() if which == 'first' else actions['time'].max()

        # Apply filtering based on 'where' and 'inclusive'
        if where == 'before':
            if include:
                result = group[group['time'] >= ref_time]
            else:
                result = group[group['time'] > ref_time]
        else:  # where == 'after'
            if include:
                result = group[group['time'] <= ref_time]
            else:
                result = group[group['time'] < ref_time]

        if include:
            at_ref_time = result['time'] == ref_time
            not_target_action = result['action'] != action
            # rows that are exactly at ref_time but not the target action
            mask_bad_same_time = at_ref_time & not_target_action
            # drop them
            result = result[~mask_bad_same_time]

        return result

    except Exception as e:
        print(f"Error in remove_before_or_after_action: {e}")
        return group
    
def add_synthetic_event(
    group: pd.DataFrame,
    action: str,
    where: ActionWhich
) -> pd.DataFrame:
    
    if (group["action"] == action).any():
        return group

    if where == "first":
        ref_time = group["time"].min()
    else:
        ref_time = group["time"].max()

    visitor_id = group['visitor_id'].iloc[0]
    # build the synthetic row
    synth_row = {
        **{col: None for col in group.columns},  # default None for all cols
        "visitor_id": visitor_id,
        "action": action,
        "time": ref_time,
    }
    synth_df = pd.DataFrame([synth_row])

    if where == "first":
        # put synthetic row first
        group = pd.concat([synth_df, group], ignore_index=True)
    else:
        # put synthetic row last
        group = pd.concat([group, synth_df], ignore_index=True)
    return group

### RAW VISUALIZATION
# CATEGORIZE ACTIONS
regex_category_map = {
    "Survey": [
        r"toggle_",
        r"survey",
        r"BackSurvey",
        r"Next(Survey|PostSurvey)",
        r"Button_Continue_(?!.*(closeCosent|openCosent))"
    ],
    "Start": [
        # r"panelReset",
        # r"open_instructions",
        r"close_Instructions",
    ],
    "Finish": [
        r"Finish_virtualNavigation"
    ],
    "Naviagation": [
        r"timeline_",
        r"UI_.*Map",
        r"UI_(BirdView|TopView|FieldView|CompassCamera|CenterCamera|joysticks)",
        r"SelectMapLocation",
    ],
    "Settings": [
        r"SelectLanguage",
    ],
    "Content": [
        r"CTRL_",
        r"ClickMouse",
        r"Exhibit_",
        r"MenuExhibitButton",
        r"OpenContent_DB_",
        r"Touch_PageLabel_",
        r"Button_Continue_(closeCosent|openCosent)",
        r"HideContent_Button"
    ]
}

# Function to assign a category based on regex patterns
def categorize_action(action: str) -> str:
    for category, patterns in regex_category_map.items():
        for pattern in patterns:
            if re.search(pattern, action):
                return category
    return "Touch"

# PLOT FUNCTION RASTER WITH CATEGORY DIMENSION AND START END
# Function to generate a scrollable HTML report (referencing image paths, not embedding them)
def generate_scrollable_html_report(
    image_paths: list[tuple[str, str]],
    output_dir: Path,
    filename: str = "report_raster.html",
    title: str = "Raster Report"
) -> Path:
    """
    Generate an HTML report displaying a horizontally scrollable list of images per visitor.

    Parameters:
        image_paths (list of tuples): Each tuple is (label, relative_image_path).
        output_dir (Path): Directory to save the HTML file.
        title (str): Title for the HTML report.

    Returns:
        Path: Path to the saved HTML file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    html_lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        f"<title>{title}</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; padding: 20px; }",
        ".visitor-section { margin-bottom: 40px; }",
        ".image-container {",
        "    overflow-x: auto;",
        "    white-space: nowrap;",
        "    border: 1px solid #ccc;",
        "    padding: 10px;",
        "}",
        "img {",
        "    height: 200px;",  # Adjust height as needed
        "    width: auto;",
        "    display: inline-block;",
        "    margin-right: 10px;",
        "}",
        "h2 { margin-top: 0; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>"
    ]

    # Group images by visitor
    from collections import OrderedDict

    grouped_images = OrderedDict()
    for visitor, fname in image_paths:
        if visitor not in grouped_images:
            grouped_images[visitor] = []
        grouped_images[visitor].append(fname)

    for visitor, files in grouped_images.items():
        html_lines.append("<div class='visitor-section'>")
        html_lines.append(f"<h2>Visitor: {visitor}</h2>")
        html_lines.append("<div class='image-container'>")
        for fname in files:
            html_lines.append(f'<img src="../{fname}" alt="Visitor {visitor}">')
        html_lines.append("</div></div>")

    html_lines.append("</body></html>")

    output_path = output_dir / filename
    with open(output_path, "w") as f:
        f.write("\n".join(html_lines))

    return output_path

# HMTL REPORT


### Interaction Processing
def load_cache(path: str):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache, path: str):
    with open(path, "w") as f:
        json.dump(cache, f)


### DEBUG

def num_start_end(df, verbose=True):
    # know ehter each visitor has start end, and ends are not duplicated
    start_event = 'Button_close_Instructions'
    end_event = 'Finish_virtualNavigation'

    start_num = df[df['action'] == start_event]['visitor_id'].nunique()
    start_num_tot = df[df['action'] == start_event].shape[0]

    end_num = df[df['action'] == end_event]['visitor_id'].nunique()
    end_num_tot = df[df['action'] == end_event].shape[0]

    if verbose:
        print(f"Number of start events: {start_num}, {start_num_tot}")
        print(f"Number of close events: {end_num}, {end_num_tot}")
