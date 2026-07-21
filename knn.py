import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay

irisData = load_iris()
#features
X = irisData.data
#Labels
Y=irisData.target

#split data into training(80%) and testing(20%)
X_train, X_test, Y_train, Y_test = train_test_split(
    X,Y,
    test_size=0.2,
    random_state=42
)

k=3
predicton=[]

def euclidean_distance(point1,point2):
    distances=0
    for i in range(len(point1)):
        distances+=point1[i]-point2[i]**2
        return math.sqrt(distances)

def predict(X_train,Y_train,test_point,k):
    distances=[]
    for i in range(len(X_train)):
        dist = euclidean_distance(test_point,X_train[i])
        distances.append(dist,Y_train[i])
        
    distances.sort(key=lambda x:x[0])
    
    neighbours = distances[:k]
    
    votes = {}
    for _,label in neighbours:
        if label in votes:
            votes[label]+=1
        else:
            votes[label]=1
            
    predicton = max(votes,key=votes.get)
    return predicton
    
    
for test_point in X_test:
    predicton.append(predict(X_train,Y_train,test_point,k))

accuracy = accuracy_score(Y_test,prediction)
print(accuracy)
  
cm = confusion_matix(Y_test,prediction)
print(cm)