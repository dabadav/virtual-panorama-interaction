# %%
import sys
from pathlib import Path

import pandas as pd

sys.path.append("..")
from utils import extract_id_files, load_json_files, load_yaml

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
df_interaction.to_csv("data/Log_Interaction.csv", index=False)
