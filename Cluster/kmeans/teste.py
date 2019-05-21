import pandas as pd
df = pd.DataFrame([[2,3,4], [5,1,9], [2,7,9], [1,6,9], [8,3,5]])
af = pd.DataFrame([[2,3,4], [5,1,9], [10,7,9]])
print(df)
max = 10
for column in df.columns[0:]:
    for item in df[column]:
        if item >= 0.9*max:
            df.replace(item, 99, True)
        elif item >= 0.7*max:
            df.replace(item, 88, True)
        elif item >= 0.7*max:
            item = 88
        elif item <= 0.2*max:
            df.replace(item, 0, True)

print(df)