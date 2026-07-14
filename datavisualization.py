import pandas as pd #working with tables and datasets
import matplotlib.pyplot as plt #creating charts and graphs
import seaborn as sb #colourful and attractive

df = pd.read_csv('student_data_cleaned.csv')
print('Cleaned dataset : ')
print(df)

plt.figure(figsize=(6,5))
#count how many males and females are in the dataset
gender_count = df['Gender'].value_counts()
#create bar charts
plt.bar(gender_count.index,gender_count.values,color=['skyblue','pink'])
plt.title('Bar chart -> count of gender')
plt.xlabel('Gender')
plt.ylabel('Count')
#plt.show()

#Bivariate analysis
plt.figure(figsize=(8,6))
plt.scatter(df['Age'],df['GPA'],color='purple')
plt.title('Scatter plot -> Age vs GPA')

#Correlation heatmap
plt.figure(figsize=(8,6)) 
corr = df[['Age','GPA','Attendance','Salary']].corr()
sb.heatmap(corr,annot=True,cmap='coolwarm',center=0,linewidths=0.5)
plt.title('Heatmap (Correlation)')
plt.show()
