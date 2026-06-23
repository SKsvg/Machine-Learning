import pandas as pd

df = pd.read_csv('C:/Users/cccuser/Desktop/22csc025/student_data.csv')
"""
print("Display few raws: ")
print(df)      #original data
print(df.head())      #top few raws
print(df.tail())      #last few raws

print('About data: ',df.info())
print('Column names: ',df.columns)

print('Description: \n',df.describe())
print('Number of raws and columns: ',df.shape)
"""
# remove fully repeated raws
df1=df.drop_duplicates(inplace=True)    # in student dataset there's no same data raws
# if the same student appears multiple times,keep only a first entry
df2=df.drop_duplicates(subset=['Student_ID'],keep='first')
print('duplicates removed.')
print(df2)  