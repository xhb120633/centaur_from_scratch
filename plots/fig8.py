import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import scienceplots
import matplotlib.gridspec as gridspec
from functools import reduce

selected_tasks  = ['peterson2021using', 'ruggeri2022globalizability']

df_random = pd.read_csv('../results/all_data_random.csv')
df_random = df_random[df_random['task'].isin(selected_tasks)][['task', 'random']]

df_llama_70b = pd.read_csv('../results/all_data_unsloth-Meta-Llama-3.1-70B-bnb-4bit.csv')
df_llama_70b = df_llama_70b[df_llama_70b['task'].isin(selected_tasks)][['task', 'unsloth/Meta-Llama-3.1-70B-bnb-4bit']]

df_centaur_70b = pd.read_csv('../results/all_data_marcelbinz-Llama-3.1-Centaur-70B-adapter.csv')
df_centaur_70b = df_centaur_70b[df_centaur_70b['task'].isin(selected_tasks)][['task', 'marcelbinz/Llama-3.1-Centaur-70B-adapter']]

df_baseline = pd.read_csv('../results/all_data_baseline.csv')
df_baseline = df_baseline[df_baseline['task'].isin(selected_tasks)][['task', 'baseline']]

df_centaur_70b_no_history = pd.read_csv('../ceiling/results/marcelbinz-Llama-3.1-Centaur-70B-adapter.csv',  index_col=0)
df_centaur_70b_no_history = df_centaur_70b_no_history.rename(columns={'marcelbinz/Llama-3.1-Centaur-70B-adapter': 'marcelbinz/Llama-3.1-Centaur-70B-adapter-no-history'})
df_centaur_70b_no_history['task'] = df_centaur_70b_no_history['task'].str.replace('/prompts_zeroshot.jsonl','')

df_llama_70b_no_history = pd.read_csv('../ceiling/results/unsloth-Meta-Llama-3.1-70B-bnb-4bit.csv', index_col=0)
df_llama_70b_no_history = df_llama_70b_no_history.rename(columns={'unsloth/Meta-Llama-3.1-70B-bnb-4bit': 'unsloth/Meta-Llama-3.1-70B-bnb-4bit-no-history'})
df_llama_70b_no_history['task'] = df_llama_70b_no_history['task'].str.replace('/prompts_zeroshot.jsonl','')

df_ceiling = pd.read_csv('../ceiling/results/ceiling.csv')
df_ceiling = df_ceiling.rename(columns={'nll': 'ceiling'})
df_ceiling = df_ceiling[df_ceiling['task'].isin(selected_tasks)][['task', 'ceiling']]

df = reduce(lambda left,right: pd.merge(left,right,on=['task'], how='outer'), [df_random, df_llama_70b, df_centaur_70b, df_centaur_70b_no_history, df_llama_70b_no_history, df_ceiling, df_baseline])
print(df)

model_names = ['random', 'marcelbinz/Llama-3.1-Centaur-70B-adapter', 'unsloth/Meta-Llama-3.1-70B-bnb-4bit', 'baseline', 'ceiling', 'marcelbinz/Llama-3.1-Centaur-70B-adapter-no-history', 'unsloth/Meta-Llama-3.1-70B-bnb-4bit-no-history']

gs = gridspec.GridSpec(1, 2, width_ratios=[0.5, 0.5])
plt.style.use(['nature'])
fig = plt.figure(figsize=(7.08661, 1.9))
for task_index, task in enumerate(df['task']):
    print(task)
    df_task = df[df['task'] == task][model_names].values.flatten()
    ll_random = -df_task[0]
    df_task = 1 - (-df_task[1:]/ll_random)
    df_task[df_task != df_task] = 0
    print(df_task)
    ax = fig.add_subplot(gs[:, task_index])

    ax.bar(np.arange(6), df_task, color=['#69005f', '#ff506e', '#cbc9e2', 'black', '#69005f', '#ff506e'])
    ax.set_xticks(np.arange(6), ['Centaur', 'Llama', 'Cognitive\nmodel', 'Noise\nceiling', 'Centaur\n(ind.)', 'Llama\n(ind.)'])
    ax.set_ylim(-0.0, 0.4)
    if task_index == 0:
        ax.set_ylabel(r'Pseudo-R$^2$')
    ax.containers[0][0].set_alpha(0.8)
    ax.containers[0][1].set_alpha(0.8)
    ax.containers[0][2].set_alpha(1)
    ax.containers[0][3].set_alpha(0.5)
    ax.containers[0][4].set_alpha(0.8)
    ax.containers[0][5].set_alpha(0.8)
    ax.containers[0][4].set_hatch('///')
    ax.containers[0][5].set_hatch('///')

sns.despine()
plt.tight_layout()
plt.savefig('figures/fig8.pdf', bbox_inches='tight')
plt.show()
