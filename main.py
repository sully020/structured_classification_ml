""" 
Supervised Learning Analysis - main.py
Christopher D. Sullivan
Professor Brian O'Neill
2/27/23

The goal of this program is to use four supervised machine learning techniques 
for two binary classification problems each. This is achieved by instantiating each technique
using sklearn, then training and testing them against the chosen data using cross-validation.
From there, average accuracy scores during test runs as well as training runs
are recorded, as well as training times in order to compare the various algorithms.

Imports:
https://pandas.pydata.org/ - pandas
https://seaborn.pydata.org/ - seaborn
https://numpy.org/          - numpy
https://matplotlib.org/     - matplotlib
https://scikit-learn.org/stable/ - scikit learn

Datasets:
https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv : Balaka Biswas
   - Classifies 5,000+ e-mails on whether or not they are considered 'spam'.

https://www.kaggle.com/code/mathchi/diagnostic-a-patient-has-diabetes/notebook : Mehmet Akturk
   - Classifies 700+ patients on whether or not they test + for diabetes.   

Credits:
https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
   - Aided in realizing visualization of confusion matrix through seaborn.
"""

import pandas as pan
import seaborn as sea
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Convert the emails.csv file to a pandas dataframe for usability.
def create_emails_dframe():
    emails = pan.read_csv('emails.csv')
    x = emails.drop(columns = ['Email No.', 'class_val'])
    y = emails['class_val']
    return x, y

# Prompts the user for a string and selects learning technique accordingly.
def choose_email_learner(choice):
    match choice:
        case 'neural network':
            learner = MLPClassifier(max_iter = 75, random_state = 0)
        case 'decision tree':
            learner = DecisionTreeClassifier(class_weight = 'balanced', random_state = 2)
        case 'k neighbors':
            learner = KNeighborsClassifier(20, weights = 'distance', p = 1)
        case 'random forest':
            learner = RandomForestClassifier(class_weight = 'balanced', random_state=2)
        case _:
            raise ValueError("Inappropriate argument value of " + str(choice) + " entered.")

    return learner

# Returns the average training & test scores as well as train time. 10-fold cross-validation.
def score_email_predictions(learner):
    x, y = create_emails_dframe()
    scores = cross_validate(learner, x, y, cv = 10, return_train_score = True)                      
    return scores

# Creates a visualized confusion matrix based on the chosen learner.
def generate_email_heatmap(learner):
    x, y = create_emails_dframe()
    preds = cross_val_predict(learner, x, y, cv = 10) 
    cf_matrix = confusion_matrix(y, preds)    
    sea.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap=('Greens'))
    plt.show()

# Output the results of training and testing the learner using emails.csv, optionally produce heatmap visualization.
def model_learning_on_emails():
    choice = input("Please type your choice of learning model for the spam e-mails dataset: \
1.) 'decision tree' 2.) 'neural network' 3.) 'k neighbors' 4.) 'random forest': ")
    learner = choose_email_learner(choice)
    email_scores = score_email_predictions(learner)

    print("The mean accuracy of the " + choice + " used to classify spam e-mails by " 
    + "10-fold cross-validation is: " + "{:.2f}".format(email_scores['test_score'].mean() * 100) + "%\n")

    print("The mean accuracy of the " + choice + " during training was: " + "{:.2f}".format(email_scores['train_score'].mean() * 100) + "%\n")

    print("The time on average that this " + choice + " took to train was " + "{:.2f}".format(email_scores['fit_time'].mean()) + " second(s).\n")

    #generate_email_heatmap(learner)



# Convert the diabetes.csv file to a pandas dataframe for usability.
def create_patients_dframe():
    patients = pan.read_csv('diabetes.csv')
    x = patients.drop(columns = 'class_val')
    y = patients['class_val']
    return x, y

# Returns the average training & test scores as well as train time. 
def choose_patients_learner(choice):
    cvk = 5
    match choice:
        case 'neural network':
            learner = MLPClassifier(67, learning_rate_init = .04, random_state = 1)
            cvk = 10
        case 'decision tree':
            learner = DecisionTreeClassifier(max_depth = 5, max_leaf_nodes = 21, random_state = 2)
        case 'k neighbors':
            learner = KNeighborsClassifier(24, weights = 'distance', p = 1)
        case 'random forest':
            learner = RandomForestClassifier(random_state = 0)
        case _:
            raise ValueError("Inappropriate argument value of " + str(choice) + " entered.")

    return learner, cvk

#  Returns the training & test scores as well as train time. k-fold cross-validation.
def score_patients_predictions(learner, cvk):
   x, y = create_patients_dframe()
   scores = cross_validate(learner, x, y, cv = cvk, return_train_score = True)                      
   return scores

# Creates a visualized confusion matrix based on the chosen learner.
def generate_patients_heatmap(learner, cvk):
    x, y = create_patients_dframe()
    preds = cross_val_predict(learner, x, y, cv = cvk)
    cf_matrix = confusion_matrix(y, preds)    
    sea.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Purples') 
    plt.show()

# Output the results of training and testing the learner using diabetes.csv, optionally produce heatmap visualization.
def model_learning_on_patients():
    choice = input("Please type your choice of learning model for the diabetes patients dataset: \
1.) 'decision tree' 2.) 'neural network' 3.) 'k neighbors' 4.) 'random forest': ")

    learner, cvk = choose_patients_learner(choice)
    patients_scores = score_patients_predictions(learner, cvk)

    print("The mean accuracy of the " + choice + " used to classify diabetes by " \
    + str(cvk) + " fold cross-validation is: " + "{:.2f}".format(patients_scores['test_score'].mean() * 100) + "%\n")

    print("The mean accuracy of the " + choice + " during training was " + "{:.2f}".format(patients_scores['train_score'].mean() * 100) + "%\n")

    print("The time on average that this " + choice + " took to train was " + "{:.2f}".format(patients_scores['fit_time'].mean()) + " second(s).")

    #generate_patients_heatmap(learner, cvk)



model_learning_on_emails()
model_learning_on_patients()