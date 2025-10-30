# %%
import os
import sys
from pathlib import Path

import pandas as pd
    
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

sys.path.append("..")
from utils import extract_id_files, load_json_files, load_yaml, num_start_end

SAVE_DATA = True

# %%
########### Load and Process
data_dir = Path("../data/Bergen-Belsen-Panel6/LogsBergenBelsen")
pattern = "Log_Interactions*.json"  # Survey files pattern
output_directory = Path("data/")
output_filename = "Log_Interaction.csv"

# Load JSONs into Pandas df
interaction = load_json_files(data_dir, pattern)
visitor_ids = extract_id_files(data_dir, pattern)
interaction_tables = [pd.json_normalize(s, sep="_") for s in interaction]
# Attach user_id to each DataFrame
for df, uid in zip(interaction_tables, visitor_ids):
    df["visitor_id"] = uid

# Combine all into one DataFrame
df_interaction = pd.concat(interaction_tables, ignore_index=True)

print(f"Number visitor_ids: {df_interaction['visitor_id'].nunique()}")
print(f"Number events: {df_interaction.shape[0]}")

if SAVE_DATA:
    df_interaction.to_csv("data/Log_Interaction.csv", index=False)

# Remove Interactions corresponding to test files
test_ids = ['1744809723872', '1744015598515', '1733218842453', '1745332528203', '1744015689042', '1744896993937', '1744810157841', '1744897044784', '1742912285792', '1744899341160', '1745335256765', '1744899669470',       '1744896736038', '1744810101012', '1744810778970']
df_interaction = df_interaction[~df_interaction['visitor_id'].isin(test_ids)]
df_interaction["time"] = pd.to_numeric(df_interaction["time"], errors='coerce')

print(f"Number visitor_ids: {df_interaction['visitor_id'].nunique()}")
print(f"Number events: {df_interaction.shape[0]}")
num_start_end(df_interaction)


# %%
########### Remove before start signal and after end signal
from utils import remove_before_or_after_action, sample, plot_sample

start_signal = 'Button_close_Instructions'
end_signal = 'Finish_virtualNavigation'

# Trim survey interaction from first Close Instruction event
df_result = df_interaction.groupby('visitor_id', group_keys=False).apply(remove_before_or_after_action, action=start_signal, which='first', where='before', include=True, fallback=False)
df_clean = df_result.groupby('visitor_id', group_keys=False).apply(remove_before_or_after_action, action=end_signal, which='last', where='after', include=True, fallback=True)

print(f"Number visitor_ids: {df_clean['visitor_id'].nunique()}")
print(f"Number events: {df_clean.shape[0]}")
num_start_end(df_clean)
# %%
########### Remove: if no end signal then trim after N minutes inactivity
from utils import remove_inactivity

# Trim after inactivity to isolate user from survey
MINUTES = 2
time_inactivity = 60 * MINUTES
df = df_clean.copy()

df["time"] = pd.to_numeric(df["time"], errors='coerce')
df["time_from_session_start"] = df["time"] - df.groupby("visitor_id")["time"].transform("min")
df["time_between_action"] = df.groupby("visitor_id")["time"].diff().fillna(0)
df["inactive"] = df["time_between_action"] >= time_inactivity
affected_visitors = df[df["inactive"]]["visitor_id"].unique()
num_start_end(df)

# Apply per user
trimmed_df = df.groupby("visitor_id", group_keys=False).apply(remove_inactivity)
print(f"Number visitor_ids: {trimmed_df['visitor_id'].nunique()}")
print(f"Number events: {trimmed_df.shape[0]}")
num_start_end(trimmed_df)

# %%
######### Add synthetic end signal to flag session end
from utils import add_synthetic_event
df_final = trimmed_df.groupby('visitor_id', group_keys=False).apply(add_synthetic_event, action=end_signal, where='last')
num_start_end(df_final)

# %%
########### Save results
df_final = df_final[["visitor_id", "action", "positionScreen", "time", "time_from_session_start", "time_between_action"]]
df_final.to_csv(
    f"data/Log_Interaction_Inactivity_{MINUTES}min.csv", index=False
)
# %%
########### Load Survey and Interaction

# Load data and filter the interaction logs from answered surveys
df_interaction = pd.read_csv(f"data/Log_Interaction_Inactivity_{MINUTES}min.csv")
df_survey = pd.read_csv("data/Log_Survey_Translated.csv", index_col=0)
visitors = df_survey.index.tolist() # Get visitor ids

# Filter just the answered surveys
df = df_interaction[df_interaction["visitor_id"].isin(visitors)]

print(f"Number visitor_ids: {df['visitor_id'].nunique()}")
print(f"Number events: {df.shape[0]}")

if SAVE_DATA:
    df.to_csv(
    f"data/Log_Interaction_Survey_Inactivity_{MINUTES}min.csv", index=False
)
