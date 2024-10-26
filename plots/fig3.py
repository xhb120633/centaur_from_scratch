import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import scienceplots
import matplotlib.gridspec as gridspec

df =  pd.read_csv('../results/fig3_data.csv')
print(df)
gs = gridspec.GridSpec(1, 3, width_ratios=[0.3333, 0.3333, 0.3333])
model_names = ['random_nll', '/home/aih/marcel.binz/Centaur-3.1/1_finetuning/centaur2-final-llama/checkpoint-2000/', 'unsloth/Meta-Llama-3.1-70B-bnb-4bit', 'baseline_nll']

plt.style.use(['nature'])
fig = plt.figure(figsize=(7.08661, 1.9))
for task_index, task in enumerate(df['task']):
    print(task)
    scale = 1 if task_index == 1 else 0.5
    df_task = df[df['task'] == task][model_names].values.flatten()
    ll_random = -df_task[0]
    df_task = 1 - (-df_task[1:]/ll_random)
    df_task[df_task == 1] = 0
    print(df_task)
    ax = fig.add_subplot(gs[:, task_index])

    ax.bar(np.arange(3), df_task, color=['#69005f', '#ff506e', '#cbc9e2'])
    ax.set_xticks(np.arange(3), ['Centaur', 'Llama', 'Cognitive\nmodel'])

    if task_index == 2:
        ax.text(0.775, 0.15, 'N/A', transform=ax.transAxes, va='top')  # Add label (c)
    ax.containers[0][0].set_alpha(0.8)
    ax.containers[0][1].set_alpha(0.8)
    ax.containers[0][2].set_alpha(1)

sns.despine()
plt.tight_layout()
plt.savefig('figures/fig3.pdf', bbox_inches='tight')
plt.show()
