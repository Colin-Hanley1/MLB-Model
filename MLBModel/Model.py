from kenpompy.utils import login
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt     

browser = login("colinhanley1836@gmail.com", "Colin0829!")

import kenpompy.summary as kp

#X, y = make_classification(n_samples=500, n_features=2,n_classes=2,n_informative=2,n_redundant=0,n_clusters_per_class=1,random_state=100   )
X = np.genfromtxt('data.csv', delimiter=',')
y = np.genfromtxt('WL.csv', delimiter=',')
#X = np.array([[10,5.5,5.9],[-9.7,-7.9,-6.2],[3.6,1.5,0.7],[-4,-0.1,-4],[-.1,2,1.9],[13.6,2,2.8],[13.1,-5.4,1.1],[13.2,-13.4,0.6],[22,-12.7,9.1],[16.6,-1.9,5.7],[.7,-.9,-4.3],[-3.1,-1.3,-1.4],[14.1,-3.1,5.9],[-7,1.8,-3.5]])
#y = np.array([1,1,1,1,0,0,1,1,1,1,0,0,0,0])   
plt.figure(figsize=(7,7))
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float().reshape(len(y), 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999)

class Logistic_Regression(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.layer0 = nn.Linear(in_features=num_features, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer0(x)
        x = self.sigmoid(x)
        return x
    
model = Logistic_Regression(num_features=10)

LEARNING_RATE = 0.001
EPOCHS = 40000
loss_function = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

def calculate_accuracy(preds, actuals):
    
    with torch.no_grad():
        rounded_preds = torch.round(preds)
        num_correct = torch.sum(rounded_preds == actuals)
        accuracy = num_correct/len(preds)
        
    return accuracy

train_losses = []
test_losses  = []
train_accs = []
test_accs  = []

for epoch in range(EPOCHS):
    # Forward propagation (predicting train data) #a
    train_preds = model(X_train)
    train_loss  = loss_function(train_preds, y_train)
    
    # Predicting test data #b
    with torch.no_grad():
        test_preds = model(X_test)
        test_loss  = loss_function(test_preds, y_test)
        
    # Calculate accuracy #c
    train_acc = calculate_accuracy(train_preds, y_train)
    test_acc  = calculate_accuracy(test_preds, y_test)
    
    # Backward propagation #d
    optimizer.zero_grad()
    train_loss.backward()

    # Gradient descent step #e
    optimizer.step()
    
    # Store training history #f
    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())
    train_accs.append(train_acc.item())
    test_accs.append(test_acc.item())
    
    # Print training data #g
    if epoch%100==0:
        print(f'Epoch: {epoch} \t|' \
            f' Train loss: {np.round(train_loss.item(),3)} \t|' \
            f' Test loss: {np.round(test_loss.item(),3)} \t|' \
            f' Train acc: {np.round(train_acc.item(),2)} \t|' \
            f' Test acc: {np.round(test_acc.item(),2)}')

with torch.no_grad():
    param_vector = torch.nn.utils.parameters_to_vector(model.parameters())





PATH = "nrfimodel.pt"
model_scripted = torch.jit.script(model)
model_scripted.save(PATH)

tester = np.array([.89,.275,.105,-.145,-.5,.018,.0415,.0155,.01,.0075])
tester = torch.from_numpy(tester).float()
print(float(model(tester)))

