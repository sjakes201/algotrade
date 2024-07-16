import pandas as pd

df = pd.read_csv('backtest_results_closing_1.csv')
for i in range(len(df)):
    row = df.iloc[i]
    text = row[0].split('Realized ')
    text2 = text[1].split(' ')[:4]
    profit = float(text2[1][1:])
    df.at[i, 'profit'] = profit

print(df['profit'])

df.to_csv('backtest_results_closing_1_fixed.csv', index=False)