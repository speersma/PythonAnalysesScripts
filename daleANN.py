"""Functions for training and testing an artificial neural network on DALE data"""

# NECESSARY IMPORTS #

import torch
import pandas as pd
from torchvision import transforms
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from torchsummary import summary

# FUNCTION FOR BUILDING ANN FEATURES
def buildNet(numFeatures, layer1Nodes, layer2Nodes, layer3Nodes, dropoutRate):
    """This function creates a class for the neural network. Some features of the
    network's architecture are defined in the inputs to this function
    Inputs:
        - numFeatures: this is the number of input variables/features
        - layer1Nodes: number of nodes in the first hidden layer
        - layer2Nodes: number of nodes in the second hidden layer
        - layer3Nodes: number of nodes in the third hidden layer
        - dropoutRate: percentage, expressed in decimal form, of nodes to be dropped
    Returns:
        - a classifier object (this net is used in the training function)"""
    class Classifer(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(numFeatures, layer1Nodes)
            self.dropout = nn.Dropout(p=dropoutRate)
            self.fc2 = nn.Linear(layer1Nodes, layer2Nodes)
            self.fc3 = nn.Linear(layer2Nodes, layer3Nodes)
            self.fc4 = nn.Linear(layer3Nodes, 1)

        def forward(self,x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = torch.sigmoid(self.fc4(x))
            return x

    net = Classifer()

    return net
# FUNCTION FOR TRAINING MODEL

def train_ANN(networkObj, X_train, y_train, learningRate=0.1, epochNum = 1000, momentm=0.5):
    """This function trains an artificial neural network based on inputs.
    Inputs:
        - networkObj: networks object generated by the buildNet function
        - X_train: X training data. (torch.float32)
        - y_train: y training data. (torch.float32)
            - y_train data must be reshaped using y_train.reshape(len(y_train),1)
        - learningRate: learning rate of the model (float)
        - epochNum: number of iterations the model should run (int)
        - momentm: momentum of gradient descent (float)
    returns:
        - network object post-training"""
    # TRAINING MODEL

    net = networkObj
    optimizer = optim.SGD(net.parameters(), lr=learningRate, momentum=momentm)
    criterion = nn.BCELoss()

    net.train()

    for e in range(epochNum):
        optimizer.zero_grad()
        pred = net(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

        print('Epoch: {} \t Training loss: {}'.format(e,loss))

    return net
# FUNCTION FOR EVALUATING MODEL

def evaluate_ANN(net, X_test, y_test):
    """This function evaluates the accuracy of the ANN by generating predicted
    probabilities using the trained network. These probabilities are then assigned
    a class based on their probabilities. Probabilties greater than .5 are assigned
    as a 1, else they are assigned class 0. The accuracy is then calculated by 
    dividing the total number of predictions that match y_test by the total number
    of predictions.
    Inputs: 
        - net: network object generated by the training function (train_ANN)
        - X_test: X testing data. (torch.float32)
        - y_test: y testing data. (torch.float32)
            - y_test data must be reshaped using y_test.reshape(len(y_test),1)
    Returns: 
        - corrects/len(pred_prob): accuracy of model (defined above in docstring)
        - pred_class: the predicted class for X_test"""
    net.eval()
    pred_prob = net(X_test)
    corrects = 0 
    pred_class = []
    for i in range(len(pred_prob)):
        if pred_prob[i] > 0.5:
            pred_class.append(1)
        else: 
            pred_class.append(0)
        if pred_class[i]==y_test[i]:
            corrects += 1
    return corrects/len(pred_prob), pred_class
