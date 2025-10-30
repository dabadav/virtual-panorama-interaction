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
from event_processor import SessionFSM

# %%
########### Load Preprocessed data
from utils import num_start_end

df = pd.read_csv("data/Log_Interaction_Inactivity_2min.csv")
print(f"Number of unique visitors: {df["visitor_id"].nunique()}")
print(f"Number of unique visitors: {df.shape[0]}")
num_start_end(df)

session_data = df[["visitor_id", "action", "time"]]
colname_map = {
    "visitor_id": "VisitorID",
    "action": "Action",
    "time": "Timestamp"
}
session_data = session_data.rename(colname_map, axis=1)
# %%
########### Extract Events
event_processor = SessionFSM(session_data)
df_sessions, df_actions = event_processor.generate_session_dataframe()

# %%
########### Prepare dataset
features = ["VISITOR_ID", "SESSION_DURATION", "TYPE", "EVENT_DURATION", "ACTIONS_COUNT", "ACTION", "ITEM_ID", "ACTION_TIMESTAMP", "ACTION_DURATION"]
data = df_actions[features]
# compute time from start from ACTION_TIMESTAMP per visitor curr row (time) - min(time)
data['TIME_FROM_START'] = data.groupby('VISITOR_ID')['ACTION_TIMESTAMP'].transform(
    lambda x: x - x.min()
)

print(f"Number of unique visitors: {data["VISITOR_ID"].nunique()}")

content_data = data[data['TYPE'] == 'CONTENT']
content_data.to_csv("data/Log_Interaction_Content_Events.csv", index=False)

print(f"Number of unique visitors: {content_data["VISITOR_ID"].nunique()}")

# %%
########### Put default item per exhibition
import json
import os
import re
import sys
sys.path.append("../ai-engine/omeka-tools")
from utils import load_cache, save_cache
from client import OmekaClient
import omeka_extractor as oe
from omeka_extractor import get_default_item

CACHE_PATH = "exhibit_default_item_cache.json"

client = OmekaClient()

def parse_exhibit_id(action_str):
    m = re.match(r"^OpenContent_DB_ExhibitID_(\d+)_Page_\d+$", str(action_str))
    if m:
        return m.group(1)
    return None

# load cache before apply
default_item_cache = load_cache(CACHE_PATH)

# # for each row  ['action']
# # parse str to get exhibit id
# # return default item_id
# # set as value to ['item_id']
def assign_item_id(row):
    exhibit_id = parse_exhibit_id(row["ACTION"])

    if exhibit_id:
        # fetch or reuse
        if exhibit_id not in default_item_cache:
            pages = client.get_exhibit_pages(exhibit_id)
            default_item_cache[exhibit_id] = get_default_item(pages)

        return default_item_cache[exhibit_id]

    # not an exhibit-open row
    return row.get("ITEM_ID", None)

# run fill
content_data["ITEM_ID"] = content_data.apply(assign_item_id, axis=1)

# persist cache after apply
save_cache(default_item_cache, CACHE_PATH)



# %%
############## Now we want to sum the zoom durations to the corresponding item_id
import re
import pandas as pd

def rollup_zoom_and_drop(df, action_col="ACTION", duration_col="ACTION_DURATION", zoom_token="UI_OpenZoomImage_Button"):
    df = df.sort_index().copy()

    last_non_zoom_idx = None
    dur_idx = df.columns.get_loc(duration_col)

    for i in range(len(df)):
        action = str(df.iloc[i][action_col])

        if zoom_token in action:
            if last_non_zoom_idx is not None:
                df.iloc[last_non_zoom_idx, dur_idx] += df.iloc[i][duration_col]
        else:
            last_non_zoom_idx = i

    # drop all zoom rows
    df = df[~df[action_col].str.contains(zoom_token, na=False)].copy()
    return df

content_data = rollup_zoom_and_drop(content_data)
item_data = content_data[['VISITOR_ID', 'ITEM_ID', 'ACTION_DURATION', 'TIME_FROM_START', 'SESSION_DURATION']].copy()

# remove NA, as some item id is unstandarized CTRL_IMG_Large2153
item_data = item_data.dropna()

item_data['VISITOR_ID'] = item_data['VISITOR_ID'].astype(int)
item_data['VISITOR_ID'] = item_data['VISITOR_ID'].astype('category')
item_data['ITEM_ID'] = item_data['ITEM_ID'].astype(int)
item_data['ITEM_ID'] = item_data['ITEM_ID'].astype('category')
item_data.to_csv("data/Log_Interaction_Item_Events.csv", index=False)

# %%
item_data = pd.read_csv("data/Log_Interaction_Item_Events.csv")

############## Add features of user / item
item_list = item_data['ITEM_ID'].unique().tolist()

###### Gather Item Features
def get_item_features(item_id):
    metadata_to_match = [
        'Translated Title (English)',
        'Translated Full Text Fragment (English)',
        'Main Caption (English)',
        'Additional Caption 1 (English)',
        'Creator',
        'Source Reference'
    ]
    
    item_metadata = oe.extract_metadata(oe.filter_json(client.get_item(item_id)))

    item_type = item_metadata['item_type__name'].unique()[0]
    if item_type == 'Still Image':
        item_type = 'IMG'
    else:
        item_type = 'TXT'

    mask = item_metadata['element__name'].isin(metadata_to_match)
    item_text = item_metadata[mask]['text'].copy()
    item_length = item_text.str.len()

    return {
        'ITEM_ID': item_id,
        'ITEM_MEDIUM': item_type,
        'ITEM_LENGTH': int(item_length.sum()),
    }

item_metadata = []
for item in item_list:
    item_metadata.append(get_item_features(item))

item_features = pd.DataFrame(item_metadata)
item_features.to_csv("data/Log_Item_Features.csv", index=False)
# if Still_Image else Text

# Item length
# Hierachy to get the text from metadata
# Count number of words / characters

# Item topic
# ?????

############## Learn a model dwell_time / item length ~ item medium, visitor_characteristics
