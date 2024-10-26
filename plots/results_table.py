import pandas as pd
import math

df = pd.read_csv('../results/fig2_data.csv')

df = df.rename(columns={"baseline_nll": "Cognitive model"})
df = df.rename(columns={"/home/aih/marcel.binz/Centaur-3.1/1_finetuning/centaur2-final-llama/checkpoint-2000/": "Centaur"})
df = df.rename(columns={"unsloth/Meta-Llama-3.1-70B-bnb-4bit": "Llama"})
df = df.rename(columns={"random_nll": "Random"})

print(df)

prompt = '\\begin{table}[]\n'
prompt += '\\centering \n'
prompt += '\\begin{tabular}{@{}lccc@{}} \n'
prompt += '\\toprule  \n'
prompt += "\\textbf{Experiment} & \\textbf{Centaur} & \\textbf{Llama} & \\textbf{Cognitive model} \\\\ \n"
prompt += '\\midrule \n'
for i, row in  df.iterrows():
    prompt += str(row['task_name']) + ' & ' + str(format(row['Centaur'], '.4f')) + ' & ' + str(format(row['Llama'], '.4f')) + ' & ' + str(format(row['Cognitive model'], '.4f')) + ' \\\\ \n'
prompt += '\\bottomrule \\\\ \n'
prompt += '\\end{tabular} \n'
prompt += '\\caption{Full log-likelihoods results on held-out participants.}\n'
prompt += '\\label{tab:tab2} \n'
prompt += '\\end{table}'
print(prompt)

df['Centaur'] = 1 - (-df['Centaur']/-df['Random'])
df['Llama'] = 1 - (-df['Llama']/-df['Random'])
df['Cognitive model'] = 1 - (-df['Cognitive model']/-df['Random'])

prompt = '\\begin{table}[]\n'
prompt += '\\centering \n'
prompt += '\\begin{tabular}{@{}lccc@{}} \n'
prompt += '\\toprule  \n'
prompt += "\\textbf{Experiment} & \\textbf{Centaur} & \\textbf{Llama} & \\textbf{Cognitive model} \\\\ \n"
prompt += '\\midrule \n'
for i, row in  df.iterrows():
    prompt += str(row['task_name']) + ' & ' + str(format(row['Centaur'], '.4f')) + ' & ' + str(format(row['Llama'], '.4f')) + ' & ' + str(format(row['Cognitive model'], '.4f')) + ' \\\\ \n'
prompt += '\\bottomrule \\\\ \n'
prompt += '\\end{tabular} \n'
prompt += '\\caption{Full pseudo-R$^2$ results on held-out participants.}\n'
prompt += '\\label{tab:tab3} \n'
prompt += '\\end{table}'

print(prompt)
