import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
from glob import glob
import scienceplots
import matplotlib.gridspec as gridspec
import scipy.stats as st
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
from functools import reduce

plot_8b = False

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

plt.style.use(['nature'])
fig = plt.figure(figsize=(7.08661, 5))

color_1 = '#69005f'  # Original Centaur (fine-tuned)
color_2 = '#ff506e'  # Pre-trained Centaur (your model)
color_3 = '#cbc9e2'  # Cognitive models

cmap1 = LinearSegmentedColormap.from_list("", ["white", color_1])
new_cmap1 = truncate_colormap(cmap1, 0.2, 1.0)
new_cmap0 = truncate_colormap(plt.get_cmap('Greys'), 0.2, 1.0)
gs = gridspec.GridSpec(3, 2, width_ratios=[0.6666, 0.3333])

# subplot 1 - Main comparison
df_exp = pd.read_csv('../experiments.csv', sep=';')

# Load your pre-trained model results (replace Llama)
df_pretrained = pd.read_csv('../results/all_data_centaur-random-init.csv')
df_pretrained = df_pretrained[df_pretrained['unseen'] == 'participants'][['task', 'centaur-random-init']]

# Load original Centaur results
df_centaur_70b = pd.read_csv('../results/all_data_marcelbinz-Llama-3.1-Centaur-70B-adapter.csv')
df_centaur_70b = df_centaur_70b[df_centaur_70b['unseen'] == 'participants'][['task', 'marcelbinz/Llama-3.1-Centaur-70B-adapter']]

# Load cognitive model baselines
df_baseline = pd.read_csv('../results/all_data_baseline.csv')
df_baseline = df_baseline[df_baseline['unseen'] == 'participants'][['task', 'baseline']]

# Load random baseline
df_random = pd.read_csv('../results/all_data_random.csv')
df_random = df_random[df_random['unseen'] == 'participants'][['task', 'random']]

# Merge all data
df = reduce(lambda left,right: pd.merge(left,right,on=['task'], how='outer'), 
           [df_pretrained, df_centaur_70b, df_baseline, df_random])

for index, row in df.iterrows():
    task_name = df_exp[df_exp['path'] == df.iloc[index]['task'] + '/']['task_name'].item()
    df.loc[index, 'task'] = task_name
df = df.groupby('task', as_index=False, sort=False).mean()

df = df[df['baseline'].notna()]
df = df.reset_index(drop=True)

# Calculate pseudo-R² for each model
ll_centaur_original = -df['marcelbinz/Llama-3.1-Centaur-70B-adapter']
ll_centaur_pretrained = -df['centaur-random-init']
ll_baselines = -df['baseline']
ll_random = -df['random']

results_centaur_original = 1 - (ll_centaur_original/ll_random)
results_centaur_pretrained = 1 - (ll_centaur_pretrained/ll_random)
results_baselines = 1 - (ll_baselines/ll_random)

print(len(results_centaur_original))

# Sort by original Centaur performance
order = np.argsort(results_centaur_original)
results_centaur_original = results_centaur_original[order]
results_centaur_pretrained = results_centaur_pretrained[order]
results_baselines = results_baselines[order]
papers = df['task'][list(order)]

print('Comparison Results:')
print(f'Original Centaur vs Cognitive models: {(results_centaur_original - results_baselines).mean():.3f}')
print(f'Pre-trained Centaur vs Cognitive models: {(results_centaur_pretrained - results_baselines).mean():.3f}')
print(f'Original vs Pre-trained Centaur: {(results_centaur_original - results_centaur_pretrained).mean():.3f}')
print(f'Original Centaur: {results_centaur_original.mean():.3f}')
print(f'Pre-trained Centaur: {results_centaur_pretrained.mean():.3f}')
print(f'Cognitive models: {results_baselines.mean():.3f}')

ax1 = fig.add_subplot(gs[:, 0])

# Create legend
custom_lines_r2 = [
    Line2D([0], [0], color=color_1, alpha=0.8, marker="o", linestyle='None', markersize=5), 
    Line2D([0], [0], color=color_2, alpha=0.8, marker="o", linestyle='None', markersize=5), 
    Line2D([0], [0], color=color_3, marker="o", linestyle='None', markersize=5)
]

# Overall performance bars
ax1.barh([len(results_centaur_original) + 1.25], [results_centaur_original.mean()],  height=0.25, color=color_1, alpha=0.8)
ax1.barh([len(results_centaur_original) + 1], [results_centaur_pretrained.mean()], height=0.25, color=color_2, alpha=0.8)
ax1.barh([len(results_centaur_original) + 0.75], [results_baselines.mean()],  height=0.25, color=color_3, alpha=1)

# Individual task performance bars
ax1.barh(np.arange(len(results_centaur_original)) + 0.25, results_centaur_original, height=0.25, color=color_1, alpha=0.8)
ax1.barh(np.arange(len(results_centaur_pretrained)), results_centaur_pretrained, height=0.25, color=color_2, alpha=0.8)
ax1.barh(np.arange(len(results_baselines)) - 0.25, results_baselines, height=0.25, color=color_3, alpha=1)

ax1.set_yticks(np.arange(len(results_centaur_original)).tolist() + [len(results_centaur_original) + 1], 
               papers.values.tolist() + ['Overall'])
ax1.set_xlabel(r'Pseudo-R$^2$')
ax1.set_xlim(0, 1)
ax1.set_ylim(-0.5, len(results_centaur_original) + 2)
ax1.legend(custom_lines_r2, ['Centaur (Fine-tuned)', 'Centaur (Pre-trained)', 'Cognitive model'], 
          frameon=False, ncols=3, bbox_to_anchor=(0.5, 1.05), loc='upper center')

# Add remaining subplots (2-4) from original fig2.py for behavioral analysis
# ... (copy subplots 2-4 from original fig2.py) ...

plt.tight_layout()
sns.despine()
plt.savefig('figures/fig2_pretrained_comparison.pdf', bbox_inches='tight')
plt.show() 