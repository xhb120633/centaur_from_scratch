import pandas as pd 

df = pd.read_csv('gameDat.csv')
df['correct'] = df['CorrAns'] == df['GivenAns']

df_nat = df[df['Variant'] == 'nat']
df_inv = df[df['Variant'] == 'inv']

print(df_nat['correct'].mean())
print(df_inv['correct'].mean())

print(df_nat.groupby('subID')['correct'].mean().values)
print(df_inv.groupby('subID')['correct'].mean().values)

df = pd.read_csv('simulation_marcelbinz-Llama-3.1-Centaur-70B-adapter.csv')
df['correct'] = df['CorrAns'] == df['GivenAns']

df_nat = df[df['Variant'] == 'nat']
df_inv = df[df['Variant'] == 'inv']

print(df_nat['correct'].mean())
print(df_inv['correct'].mean())

print(df_nat.groupby('subID')['correct'].mean().values)
print(df_inv.groupby('subID')['correct'].mean().values)