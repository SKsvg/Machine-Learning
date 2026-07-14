import pandas as pd

df = pd.read_csv('C:/Users/cccuser/Desktop/22csc025/student_data.csv')

df=df.drop_duplicates(subset=['Student_ID'],keep='first')

#fix inconsistent formatting
df['Gender']=df['Gender'].str.lower().str.strip()
gender_map={'male':'Male','female':'Female','f':'Female','m':'Male'}
df['Gender']=df['Gender'].map(gender_map)

#fix incorrect data types
df['Age']=pd.to_numeric(df['Age'],errors='coerce')

#convert to numeric and coerce - if conversion fails,force to mark the value as missing(NAN)
import numpy as np
df['Department']=df['Department'].replace('0',np.nan)

#Fill missing values
#df['Age']=df['Age'].fillna(df['Age'].median())       #here 150 can be affect if we get mean
df['Age']=df['Age'].fillna(30) 		#replace missing value using 30
df['Attendance']=df['Attendance'].fillna(df['Attendance'].mean())
df['Department']=df['Department'].fillna(df['Department'].mode()[0])       #bcz of numerical data we get mod, this function not replace numerical values like 0
#fillna - "Fill missing values." It replace NAN with another value.
#mode()[0]-first mode value

#handdle noisy data
df['Department']=df['Department'].replace('Computer Since','Computer Science')

#define the salary limit
upper_limit = 60000
lower_limit = 30000

#cap salary values
df['Salary']= np.where(
	df['Salary']>upper_limit,
	upper_limit,
	np.where(
		df['Salary']<lower_limit,
		lower_limit,
		df['Salary']
	)
)

#define the age limit
upper_limit = 22
lower_limit = 20

#cap age values
df['Age']= np.where(
	df['Age']>upper_limit,
	upper_limit,
	np.where(
		df['Age']<lower_limit,
		lower_limit,
		df['Age']
	)
)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
scaler_minmax = MinMaxScaler()
df['Attendance_Normalized'] = scaler_minmax.fit_transform(df[['Attendance']])

#Standardization (z-score) on salary (mean=0, std=1)
scaler_std =StandardScaler()
df['Salary_Standardized'] = scaler_std.fit_transform(df[['Salary']])

#label encoding
le = LabelEncoder()
df['Department'] = le.fit_transform(df['Department'])

#method B: one-hot encoding for nominal data (gender & city)
df = pd.get_dummies(df, columns=['Gender'], dtype=int)
print("\n--- 6.After categorical encoding ---")
print(df.info())
'''
df.to_csv('./datasets/student_data_cleaned.csv',index=False)
#index=False - "Do not save the extra index column."
print("\n Cleaned dataset saved as 'student_data_cleaned.csv' \n")'''
import os

os.makedirs("datasets", exist_ok=True)

df.to_csv("./datasets/student_data_cleaned.csv", index=False)

print("\nCleaned dataset saved as 'student_data_cleaned.csv'\n")