
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def train_logRegCV(X_train, y_train, regularization=10, folds=5, score='accuracy', 
maxIter=100, classWeight=None, seed=None):
    """This function trains a cross-validated logistic regression using sci-kit learn
    LogisticRegressoinCV classifier. Several hyperparameters are included in the input
    arguments to this function.
    Inputs:
        - X_train: X data (numpy array or pandas dataframe)
        - y_train: binary response/dependent variable (numpy array or pandas dataframe)
        - regularization: inverse of regularization strength (int or float)
        - folds: number of cross-validation folds (int)
        - score: scoring metric to be used (string)
        - maxIter: the maximum number of iterations of the optimization algorithm
        - classWeight: weights associated with classes 
            - (dict in the form of {class label: weight} or 'balanced')
        - seed: seed used for randomization
    Returns:
        - logit_model_CV: trained logistic regression model object
            - used as input for test_logRegCV"""
    
    # Input processing 

    ## chaning inputs to arrays
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    ## testing if dependent variable is one-dimensional
    if y_train.ndim > 1:
        raise TypeError("Dependent varible has {number} too many dimensions."
        .format(number=y_train.ndim-1))

    ## testing if the dependent variable is binary
    if np.any(y_train > 1):
        raise TypeError("Dependent variable is not binary")


    # CREATING AND RUNNING MODEL

    ## creating model
    logit_model_CV = LogisticRegressionCV(Cs=regularization, cv=folds, scoring=score,
                                            max_iter=maxIter, class_weight=classWeight, random_state=seed)

    ## training model
    logit_model_CV.fit(X_train, y_train)

    # EVALUATING MODEL
    print(logit_model_CV.score(X_train,y_train))

    # RETURNS 
    return logit_model_CV


def test_logRegCV(X, y, model, seed=None):
    """This function tests a trained cross-validated logistic regression model on testing data.
    The model is evaluated by using the trained model to make predictions based on the X_test
    data. These predictions (y_test_pred) are compared with y_test to generate an accuracy score.
    Inputs:
        - X: X testing data (
        - y: y testing data (pandas dataframe or numpy array)
        - model: trained LogisticRegressionCV model object
    Returns:
        - y_test_pred: the models predictions of reoffense (1) or no reoffense (0)
        - report: a classification report of the model"""

    # INPUT PROCESSING 
    X_test = np.asarray(X)
    y_test = np.asarray(y)

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

    ## generating confusion matrix 
    cm = confusion_matrix(y_test,y_test_pred)
    cm_display = ConfusionMatrixDisplay(cm).plot()

    ## generating classification report 
    targets = ['0 no reoffense', '1 reoffense']
    report = classification_report(y_test, y_test_pred, target_names=targets)

    # RETURNS 
    return y_test_pred, report