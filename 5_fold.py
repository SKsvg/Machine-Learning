from sklearn.model.selection import KFold
from sklearn.metrics import accuracy_score

k=3

kf=KFold(n_splits=5,shuffle=True,random_state=42)

accuracies=[]

for train_index,test_index