"""Module that contains functions for training and testing logistic regression ML model"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# FUNCTION 1 - TRAINING AND TESTING LOGISTIC REGRESSION USING TRAINING DATA AS INPUT


def train_logReg(X_train, y_train, seed=None):
    """
    This function takes the analysis variables as input.
    IMPORTANT: only input training data
    Inputs are used to train logistic regression model.
    Logit model currently uses liblinear solver.
    GridSearchCV integration planned once the hyperparameters
    are identified.
    Args:
        dv (series or 1d numpy array): binary dependent variable in model
        ivs (list of pandas series or 1d numpy arrays): independent variables in the model
        seed = seed used for reproduceability
    Returns:
        logit_model: the trained logistic regression model
        cm_display: Confusion matrix
        report: Classification report
    """

    # INPUT PROCESSING

    ## changing inputs to arrays
    y_train = np.asarray(y_train)
    X_train = np.asarray(X_train)


    ## testing if dependent variable is one-dimensional
    if y_train.ndim > 1:
        raise TypeError("Dependent varible has {number} too many dimensions."
        .format(number=y_train.ndim-1))

    ## testing if the dependent variable is binary
    if np.any(y_train > 1):
        raise TypeError("Dependent variable is not binary")

    # CREATING AND RUNNING MODEL
    ## creating model
    logit_model = LogisticRegression(solver='liblinear', random_state=seed)

    ## training model
    logit_model.fit(X_train, y_train)
    
    ## EVALUATING TRAINED MODEL
    print(logit_model.score(X_train, y_train))

    # RETURNS
    return logit_model 
    

def test_logReg(X_test, y_test, model, seed=None):

    # INPUT PROCESSING

    # Converting inputs to numpy arrays

    y_test = np.asarray(y_test)
    X_test = np.asarray(X_test)

    ## testing if dependent variable is one-dimensional
    if y_test.ndim > 1:
        raise TypeError("Dependent varible has {number} too many dimensions."
        .format(number=y_train.ndim-1))

    ## testing if the dependent variable is binary
    if np.any(y_test > 1):
        raise TypeError("Dependent variable is not binary")


    # EVALUATING MODEL ON TESTING DATA
    
    ## generating predicted values using trained model on X_test
    y_test_pred = model.predict(X_test)
    
    ## generating predicted probabilities using trained model
    y_pred_probs = model.predict_proba(X_test)[:,1]
    
    
    ## generating confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    cm_display = ConfusionMatrixDisplay(cm).plot()

    ## generating classification report
    targets = ['no reoffense', 'reoffense']
    report = classification_report(y_test, y_test_pred, target_names=targets)
    
    ## things the model returns
    return y_test_pred, y_pred_probs, report # p-values
    
    
    