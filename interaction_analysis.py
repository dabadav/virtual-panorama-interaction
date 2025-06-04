# %%
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp
import itertools

df = pd.read_csv('data/Log_Interaction_Survey_Inactivity_2min_Clean_NoFinish.csv')
df_survey = pd.read_csv("data/Log_Survey_Persona.csv", index_col=0)
df_survey.reset_index(names="visitor_id", inplace=True)
# df_survey = df_survey[df_survey['profile'] != 'Researcher']
df_info = df_survey.merge(df['visitor_id'].drop_duplicates(), on="visitor_id", how="inner")


# %%
fitem_id_to_medium = {
    2155: 'Diary / witness text',
    2154: 'Artwork',
    2153: 'Object',
    2152: 'Scanned document',
    2151: 'Object',
    2150: 'Photograph',
    2149: 'Diary / witness text',
    2148: 'Artwork',
    2147: 'Artwork',
    2146: 'Photograph',
    2126: 'Scanned document ; Diary / witness text',
    2125: 'Artwork',
    2124: 'Artwork',
    2123: 'Scanned document ; Map',
    2122: 'Artwork',
    2121: 'Artwork',
    2120: 'Photograph',
    2119: 'Object',
    2118: 'Contextual text',
    2117: 'Artwork',
    2116: 'Diary / witness text',
    2115: 'Diary / witness text',
    2114: 'Photograph',
    2113: 'Diary / witness text',
    2112: 'Contextual text',
    2111: 'Photograph',
    2110: 'Diary / witness text',
    2109: 'Location',
    2108: 'Contextual text',
    2107: 'Location',
    2106: 'Contextual text',
    2105: 'Location',
    2104: 'Artwork',
    2103: 'Location ; Contextual text',
    2102: 'Audio ; Witness account',
    2101: 'Contextual text',
    2100: 'Location',
    2099: 'Diary / witness text',
    2098: 'Contextual text',
    2097: 'Location',
    2096: 'Audio ; Witness account',
    2095: 'Artwork',
    2094: 'Location',
    2093: 'Photograph',
    2092: 'Location',
    2091: 'Contextual text',
    2090: 'Location',
    2089: 'Contextual text',
    2088: 'Location',
    2087: 'Contextual text',
    2086: 'Location',
    2085: 'Diary / witness text',
    2084: 'Contextual text',
    2083: 'Location ; Contextual text',
    2082: 'Contextual text',
    2081: 'Location ; Contextual text',
    2080: 'Diary / witness text',
    2079: 'Contextual text',
    2078: 'Location',
    2077: 'Photograph',
    2076: 'Contextual text',
    2075: 'Location',
    2074: 'Photograph',
    2073: 'Contextual text',
    2072: 'Contextual text',
    2071: 'Location',
    2070: 'Contextual text',
    2069: 'Location ; Contextual text',
    2068: 'Diary / witness text',
    2067: 'Artwork',
    2066: 'Contextual text',
    2065: 'Location',
    2064: 'Artwork',
    2063: 'Contextual text',
    2062: 'Location'
    # Add the rest of your mapping here
}

####
# Test for features distribution

# Session Time
import scipy.stats as stats

# Assume df and df_info are already loaded:
# df must include: 'visitor_id' and 'time_from_session_start' columns
# df_info must include: 'visitor_id' and 'profile' columns

# Compute session time per visitor
df['time_from_session_start'] = df['time_from_session_start'] / 60
session_time = df.groupby('visitor_id')['time_from_session_start'].max().reset_index()
session_time.columns = ['visitor_id', 'session_duration']

# Merge session time with visitor profiles
df_combined = df_info.merge(session_time, on='visitor_id', how='inner')

# Convert duration to seconds for analysis
# === Levene's Test for homogeneity of variance ===
groups = [group['session_duration'].values for _, group in df_combined.groupby('profile')]
levene_stat, levene_p = stats.levene(*groups)
print(f"Levene’s Test: W={levene_stat:.4f}, p={levene_p:.4f}")

# === ANOVA to test group differences ===
anova_stat, anova_p = stats.f_oneway(*groups)
print(f"ANOVA: F={anova_stat:.4f}, p={anova_p:.4f}")

# === Optional: Boxplot to visualize session time by profile ===
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_combined, x='profile', y='session_duration')
plt.ylabel("Session Duration (min)")
plt.title("Session Time by Profile")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Compute and display mean session time by profile
mean_by_profile = df_combined.groupby('profile')['session_duration'].mean().reset_index()
print(mean_by_profile)


# %%

# 2. Tukey's HSD test
tukey = pairwise_tukeyhsd(endog=df_combined['session_duration'],
                          groups=df_combined['profile'],
                          alpha=0.05)
print(tukey)

# Step 3 - Fix: Inspect structure first (optional)
for row in tukey._results_table.data[:2]:
    print(row)

# Correctly extract group1, group2, and p-value
significant_pairs = [(row[0], row[1]) for row in tukey._results_table.data[1:] if row[4] < 0.05]

# 4. Plot boxplot
plt.figure(figsize=(12, 8))
ax = sns.boxplot(data=df_combined, x='profile', y='session_duration')
plt.ylabel("Session Duration (seconds)")
plt.title("Session Time by Profile")
plt.xticks(rotation=45)

# 5. Annotate significant pairs
def add_stat_annotation(ax, pairs, data, x_col, y_col):
    """Adds statistical annotation to the boxplot"""
    y_max = data[y_col].max()
    step = y_max * 0.12
    current_height = y_max + step

    for a, b in pairs:
        x1, x2 = sorted([a, b], key=lambda k: data[x_col].unique().tolist().index(k))
        x1_idx = data[x_col].unique().tolist().index(x1)
        x2_idx = data[x_col].unique().tolist().index(x2)

        # Draw line
        ax.plot([x1_idx, x1_idx, x2_idx, x2_idx],
                [current_height, current_height + step / 2, current_height + step / 2, current_height],
                lw=1.5, color='black')

        # Add asterisk or "p<0.05"
        ax.text((x1_idx + x2_idx) / 2, current_height + step * 0.4, "*",
                ha='center', va='bottom', fontsize=12)
        current_height += step * 1.5

add_stat_annotation(ax, significant_pairs, df_combined, 'profile', 'session_duration')

plt.tight_layout()
plt.show()


# Number of elements
# Elements types

# Consistent within group
# Effects between group
# %%
