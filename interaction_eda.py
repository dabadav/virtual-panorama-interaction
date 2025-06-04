# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from pathlib import Path

# Load data and filter the interaction logs from answered surveys
df_interaction = pd.read_csv("data/Log_Interaction.csv")
df_survey = pd.read_csv("data/Log_Survey_Persona.csv", index_col=0)
visitors = df_survey.index.tolist() # Get visitor ids
print(df_interaction['visitor_id'].nunique())
df = df_interaction[df_interaction["visitor_id"].isin(visitors)]
print(df['visitor_id'].nunique())

# %%
#######
## Visualization
#######

from utils import categorize_action, plot_sample_by_category, sample
# Visualize interactions
df_vis = df.copy()
# df_vis = trimmed_df.copy()
df_vis['category'] = df_vis['action'].apply(categorize_action)
# Directory to store figures
figures_dir = Path("figures")
figures_dir.mkdir(parents=True, exist_ok=True)
# Generate and save figures
image_paths = []
all_cats = ["Content", "Navigation", "Survey", "Settings", "Touch"]
colors = {
    "Content": "blue",
    "Navigation": "orange",
    "Survey": "green",
    "Settings": "purple",
    "Touch": "gray"
}
from utils import plot_all_samples_consistently
# Step 1: Collect samples and filenames
unit = 'min'
samples = []
save_paths = []
image_paths = []
for visitor in df_vis["visitor_id"].unique():
    sample_df = sample(df_vis, visitor)  # Your custom function to extract per-visitor DataFrame
    filename = f"figures/visitor_{visitor}_{unit}_clean_nofinish.png"
    samples.append(sample_df)
    save_paths.append(filename)
    image_paths.append((visitor, filename))
plot_all_samples_consistently(
    samples=samples,
    save_paths=save_paths,
    unit=unit,
    all_categories=all_cats,
    category_colors=colors
)


# Create simple HTML report
from utils import generate_scrollable_html_report
from pathlib import Path

mode = 'clean'
reports_dir = Path("reports")
figures_dir = Path("figures")
pattern = f"visitor_*_min_{mode}_nofinish.png"
image_paths = [
    (img.stem.split('_')[1], img.relative_to("."))  # or img.name for filename only
    for img in figures_dir.glob(f"visitor_*_min_{mode}_nofinish.png")
]
html_report_path = generate_scrollable_html_report(image_paths, reports_dir, filename=f"report_{mode}_nofinish.html", title="Visitor Action Raster Report")
html_report_path

# Check session time distribution plot
time_spent = (
    df.groupby("visitor_id")["time"].agg(lambda x: (x.max() - x.min()) // 60).to_frame()
)
sns.histplot(time_spent[time_spent["time"] < 30])
plt.show()

# %%
from utils import remove_data_before_session_start, remove_before_or_after_action, sample, plot_sample

start_signal = 'Button_close_Instructions'
end_signal = 'Finish_virtualNavigation'

# Trim survey interaction from first Close Instruction event
# df_result = df.groupby('visitor_id', group_keys=False).apply(remove_data_before_session_start)
df_result = df.groupby('visitor_id', group_keys=False).apply(remove_before_or_after_action, action=start_signal, which='first', where='before', include=True, fallback=False)
df_clean = df_result.groupby('visitor_id', group_keys=False).apply(remove_before_or_after_action, action=end_signal, which='last', where='after', include=True, fallback=True)

# %%
from utils import plot_sample, remove_inactivity, sample
# Check sessions too long
outliers = time_spent[time_spent["time"] > 30]
outliers_ids = outliers.index.tolist()
for outlier_id in outliers_ids:
    plot_sample(sample(df, outlier_id), unit="min")

# %%
from utils import remove_inactivity
# Trim after inactivity to isolate user from survey
MINUTES = 2
time_inactivity = 60 * MINUTES

df = df_clean.copy()

df["time_from_session_start"] = df["time"] - df.groupby("visitor_id")["time"].transform("min")
df["time_between_action"] = df.groupby("visitor_id")["time"].diff().fillna(0)
df["inactive"] = df["time_between_action"] >= time_inactivity
affected_visitors = df[df["inactive"]]["visitor_id"].unique()

# Apply per user
trimmed_df = df.groupby("visitor_id", group_keys=False).apply(remove_inactivity)

# %%
# Visualize time between actions and total time
time_spent_trimmed = (
    df_result.groupby("visitor_id")["time"]
    .agg(lambda x: (x.max() - x.min()) // 60)
    .to_frame()
)

sns.histplot(time_spent_trimmed["time"])
plt.show()

sns.histplot(df_result["time_between_action"])
plt.show()


# %%

# For each visitor check count close instructions and 

# %%
# Save results
# trimmed_df.drop(columns="inactive", inplace=True)
trimmed_df.to_csv(
    f"data/Log_Interaction_Survey_Inactivity_{MINUTES}min_Clean_NoFinish.csv", index=False
)

# %%
