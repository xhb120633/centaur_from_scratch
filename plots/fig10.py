import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import matplotlib.gridspec as gridspec
import numpy as np

centaur_70b = torch.load('../generalization/results/additional_generalization_full_log_likelihoods_marcelbinz-Llama-3.1-Centaur-70B-adapter.pth')
centaur_8b = torch.load('../generalization/results/additional_generalization_full_log_likelihoods_marcelbinz-Llama-3.1-Centaur-8B-adapter.pth')
llama_70b = torch.load('../generalization/results/additional_generalization_full_log_likelihoods_unsloth-Meta-Llama-3.1-70B-bnb-4bit.pth')
llama_8b = torch.load('../generalization/results/additional_generalization_full_log_likelihoods_unsloth-Meta-Llama-3.1-8B-bnb-4bit.pth')

nll_random = {
    'additional_experiments/awad2018moral.jsonl':  -math.log(1/2),
    'additional_experiments/demircan2024evaluatingcategory.jsonl':  -math.log(1/2),
    'additional_experiments/demircan2024evaluatingreward.jsonl':  -math.log(1/2),
    'additional_experiments/akata2023repeatedgames.jsonl':  -math.log(1/2),
    'additional_experiments/singh2022representing.jsonl':  -math.log(1/7),
    'additional_experiments/xu2021novelty.jsonl':  -math.log(1/3),
}

task_names = {
    'additional_experiments/awad2018moral.jsonl': 'Moral decision-making',
    'additional_experiments/demircan2024evaluatingcategory.jsonl':  'Naturalistic category learning',
    'additional_experiments/demircan2024evaluatingreward.jsonl':  'Naturalistic reward learning',
    'additional_experiments/akata2023repeatedgames.jsonl':  'Economic games',
    'additional_experiments/singh2022representing.jsonl':  'Behavioral propensities',
    'additional_experiments/xu2021novelty.jsonl': 'Deep sequential decision task',
}

gs = gridspec.GridSpec(2, 3, width_ratios=[0.3333, 0.3333, 0.3333])
plt.style.use(['nature'])
fig = plt.figure(figsize=(7.08661, 3.8))

for i,  key in enumerate(centaur_70b.keys()):
    print(key)
    centaur_70b_r2 = (1 - (-centaur_70b[key] / -nll_random[key])).mean().item()
    centaur_8b_r2 = (1 - (-centaur_8b[key] / -nll_random[key])).mean().item()
    llama_70b_r2 = (1 - (-llama_70b[key] / -nll_random[key])).mean().item()
    llama_8b_r2 = (1 - (-llama_8b[key] / -nll_random[key])).mean().item()
    centaur_70b_r2_se = (1 - (-centaur_70b[key] / -nll_random[key])).std().item() / math.sqrt(len(centaur_70b[key]))
    centaur_8b_r2_se = (1 - (-centaur_8b[key] / -nll_random[key])).std().item() / math.sqrt(len(centaur_8b[key]))
    llama_70b_r2_se = (1 - (-llama_70b[key] / -nll_random[key])).std().item()  / math.sqrt(len(llama_70b[key]))
    llama_8b_r2_se = (1 - (-llama_8b[key] / -nll_random[key])).std().item()  / math.sqrt(len(llama_8b[key]))

    ax = fig.add_subplot(gs[0 if i < 3 else 1, i % 3])
    ax.bar(np.arange(4), np.array([centaur_70b_r2, centaur_8b_r2, llama_70b_r2, llama_8b_r2]), yerr=[centaur_70b_r2_se, centaur_8b_r2_se, llama_70b_r2_se, llama_8b_r2_se], color=['#69005f', '#69005f', '#ff506e', '#ff506e'])# 'C0', 'C1'
    ax.set_xticks(np.arange(4), ['Centaur\n(70B)', 'Centaur\n(8B)', 'Llama\n(70B)', 'Llama\n(8B)'])
    ax.set_ylim(0)
    if i == 0 or i == 3:
        ax.set_ylabel(r'Pseudo-R$^2$')
    ax.containers[1][0].set_alpha(0.8)
    ax.containers[1][1].set_alpha(0.5)
    ax.containers[1][2].set_alpha(0.8)
    ax.containers[1][3].set_alpha(0.5)
    ax.set_title(task_names[key], fontsize=8)
sns.despine()
plt.tight_layout()
plt.savefig('figures/fig10.pdf', bbox_inches='tight')
plt.show()
