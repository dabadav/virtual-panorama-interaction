# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df_interaction = pd.read_csv("data/Log_Interaction.csv")
df_survey = pd.read_csv("data/Log_Survey_Persona.csv", index_col=0)

visitors = df_survey.index.tolist()
# %%
# Check long session

df = df_interaction[df_interaction["visitor_id"].isin(visitors)]

# %%
time_spent = (
    df.groupby("visitor_id")["time"].agg(lambda x: (x.max() - x.min()) // 60).to_frame()
)

sns.histplot(time_spent[time_spent["time"] < 30])
plt.show()

# %%
from utils import plot_sample, remove_inactivity, sample

outliers = time_spent[time_spent["time"] > 30]

# %%
outliers_ids = outliers.index.tolist()
for outlier_id in outliers_ids:
    plot_sample(sample(df, outlier_id), unit="min")

# %%
MINUTES = 2
time_inactivity = 60 * MINUTES
df["time_from_session_start"] = df["time"] - df.groupby("visitor_id")["time"].transform(
    "min"
)
df["time_between_action"] = df.groupby("visitor_id")["time"].diff().fillna(0)
df["inactive"] = df["time_between_action"] >= time_inactivity
affected_visitors = df[df["inactive"]]["visitor_id"].unique()

# Apply per user
trimmed_df = df.groupby("visitor_id", group_keys=False).apply(remove_inactivity)


# %%
time_spent_trimmed = (
    trimmed_df.groupby("visitor_id")["time"]
    .agg(lambda x: (x.max() - x.min()) // 60)
    .to_frame()
)

sns.histplot(time_spent_trimmed["time"])
plt.show()

sns.histplot(trimmed_df["time_between_action"])
plt.show()
# %%
trimmed_df.drop(columns="inactive", inplace=True)
trimmed_df.to_csv(
    f"data/Log_Interaction_Survey_Inactivity_{MINUTES}min.csv", index=False
)
