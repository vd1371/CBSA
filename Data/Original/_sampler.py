import pandas as pd

df = pd.read_excel('analytics_challenge_dataset_ex210911.xlsx')
print('read excel')
df = df.iloc[:10000,:]
# print(df['content'])

df.to_csv('./sample.csv', index=False, header=True, encoding='utf-8-sig')

# df = pd.read_csv('sample.csv')
# print(df.columns)