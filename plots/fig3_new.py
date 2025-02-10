import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import scienceplots
import matplotlib.gridspec as gridspec
from functools import reduce
import torch
import math

centaur_70b = torch.load('../generalization/results/generalization_full_log_likelihoods_marcelbinz-Llama-3.1-Centaur-70B-adapter.pth')
llama_70b = torch.load('../generalization/results/generalization_full_log_likelihoods_unsloth-Meta-Llama-3.1-70B-bnb-4bit.pth')

df_baseline = pd.read_csv('../results/all_data_baseline.csv')
df_baseline = df_baseline[df_baseline['unseen'] == 'experiments'][['task', 'baseline']]

df_random = pd.read_csv('../results/all_data_random.csv')
df_random = df_random[df_random['unseen'] == 'experiments'][['task', 'random']]

means = {}
sems = {}
for key in centaur_70b.keys():
    print(key)
    print(centaur_70b[key].shape)
    baseline = df_baseline[df_baseline['task'] == key]
    random = df_random[df_random['task'] == key]
    means[key] = []
    sems[key] = []
    means[key].append(centaur_70b[key].mean())
    means[key].append(llama_70b[key].mean())
    sems[key].append(centaur_70b[key].std() / math.sqrt(len(centaur_70b[key])))
    sems[key].append(llama_70b[key].std() / math.sqrt(len(llama_70b[key])))

    if len(baseline) > 0:
        means[key].append(baseline.baseline.item())
    else:
        means[key].append(0)
    sems[key].append(0)
    means[key].append(random.random.item())
    sems[key].append(0)
    print()


#print(dfgdfgfd)
gs = gridspec.GridSpec(1, 3, width_ratios=[0.3333, 0.3333, 0.3333])

plt.style.use(['nature'])
fig = plt.figure(figsize=(7.08661, 1.9))
for task_index, task in enumerate(means.keys()):
    print(task)
    ax = fig.add_subplot(gs[:, task_index])
    ax.bar(np.arange(4), means[task], yerr=sems[task], color=['#69005f', '#ff506e', '#cbc9e2', 'grey'])
    print(means[task])
    ax.set_xticks(np.arange(4), ['Centaur', 'Llama', 'Cognitive\nmodel', 'Random'])

    if task_index == 2:
        ax.text(0.575, 0.15, 'N/A', transform=ax.transAxes, va='top')
    if task_index == 0:
        ax.set_ylabel('Negative log-likelihood')
    ax.containers[1][0].set_alpha(0.8)
    ax.containers[1][1].set_alpha(0.8)
    ax.containers[1][2].set_alpha(1)


sns.despine()
plt.tight_layout()
plt.savefig('figures/fig3_new.pdf', bbox_inches='tight')
plt.show()
