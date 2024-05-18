import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

with open('data.pickle', 'rb') as f:
    dataDict = pickle.load(f)

data = dataDict['data']
labels = dataDict['labels']

XTrain, XTest, yTrain, yTest = train_test_split(np.array(data), labels, test_size=0.15, random_state=22, shuffle=True)

models = {
    'RandomForest': RandomForestClassifier(random_state=22),
    'LogisticRegression': LogisticRegression(random_state=22),
    'SVC': SVC(random_state=22),
    'KNeighbors': KNeighborsClassifier(),
    'DecisionTree': DecisionTreeClassifier(random_state=22),
    'GaussianNB': GaussianNB()
}

for modelName, model in models.items():
    model.fit(XTrain, yTrain)
    yPred = model.predict(XTest)
    accuracy = accuracy_score(yTest, yPred)
    confMatrix = confusion_matrix(yTest, yPred)
    
    plt.figure(figsize=(6, 4))
    plt.matshow(confMatrix, cmap=plt.cm.Blues, alpha=0.7)
    for i in range(confMatrix.shape[0]):
        for j in range(confMatrix.shape[1]):
            plt.text(x=j, y=i, s=f'{confMatrix[i, j]}', va='center', ha='center')

    plt.title(f'{modelName} Confusion Matrix\nAccuracy: {accuracy:.2f}', pad=20)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.colorbar()
    plt.show()

    model_filename = f'{modelName}_model.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
