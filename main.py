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

def create_emails_dframe():
    emails = pan.read_csv('emails.csv')
    x = emails.drop(columns = ['Email No.', 'class_val'])
    y = emails['class_val']
    return x, y

def choose_email_learner(choice):
    match choice:
        case 'decision tree':
            # weights = {i: 3000 - i for i in range(3000)}  // deprecated, 'balanced' further increases accuracy.
            learner = DecisionTreeClassifier(class_weight = 'balanced')
        case 'neural network':
            learner = MLPClassifier()
        case 'k neighbors':
            learner = KNeighborsClassifier(20, weights = 'distance', p = 1)
        case 'random forest':
            learner = RandomForestClassifier(class_weight = 'balanced', random_state=2)
        case _:
            raise ValueError("Inappropriate argument value of " + str(choice) + " entered.")

    return learner

def score_email_predictions(learner):
    x, y = create_emails_dframe()
    scores = cross_validate(learner, x, y, cv = 10, return_train_score = True)                      
    return scores

def generate_email_heatmap(learner):
    x, y = create_emails_dframe()
    preds = cross_val_predict(learner, x, y, cv = 10) 
    cf_matrix = confusion_matrix(y, preds)    
    sea.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap=('YlOrRd'))
    plt.show()

def model_learning_on_emails():
    choice = input("Please type your choice of learning model for the spam e-mails dataset: \
1.) 'decision tree' 2.) 'neural network' 3.) 'k neighbors' 4.) 'random forest': ")
    learner = choose_email_learner(choice)
    email_scores = score_email_predictions(learner)

    print("The mean accuracy of the " + choice + " used to classify spam e-mails by " 
    + "10-fold cross-validation is: " + "{:.2f}".format(email_scores['test_score'].mean() * 100) + "%\n")

    print("The mean accuracy of the " + choice + " during training was: " + "{:.2f}".format(email_scores['train_score'].mean() * 100) + "%")

    generate_email_heatmap(learner)



def create_patients_dframe():
    patients = pan.read_csv('diabetes.csv')
    x = patients.drop(columns = 'class_val')
    y = patients['class_val']
    return x, y

def choose_patients_learner(choice):
    # CV 5 produces better results for some of these
    cv = 5
    match choice:
        case 'decision tree':
            learner = DecisionTreeClassifier(max_depth = 5, max_leaf_nodes = 21, random_state = 1)
            cv = 10
        case 'neural network':
            learner = MLPClassifier(67, learning_rate_init = .04, random_state = 1)
            cv = 10
        case 'k neighbors':
            learner = KNeighborsClassifier(19, weights = 'distance', p = 1) # weights and p only increase by tenths of percentage, not whole percentages like emails
        case 'random forest':
            learner = RandomForestClassifier(random_state = 0)
        case _:
            raise ValueError("Inappropriate argument value of " + str(choice) + " entered.")

    return learner, cv

def score_patients_predictions(learner, cv):
   x, y = create_patients_dframe()
   scores = cross_validate(learner, x, y, cv = cv, return_train_score = True)                      
   return scores

def generate_patients_heatmap(learner, cv):
    x, y = create_patients_dframe()
    preds = cross_val_predict(learner, x, y, cv = cv)
    cf_matrix = confusion_matrix(y, preds)    
    sea.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Oranges')
    plt.show()

def model_learning_on_patients():
    choice = input("Please type your choice of learning model for the diabetes patients dataset: \
1.) 'decision tree' 2.) 'neural network' 3.) 'k neighbors' 4.) 'random forest': ")

    learner, cv = choose_patients_learner(choice)
    patients_scores = score_patients_predictions(learner, cv)

    print("The mean accuracy of the " + choice + " used to classify diabetes by " \
    + str(cv) + " fold cross-validation is: " + "{:.2f}".format(patients_scores['test_score'].mean() * 100) + "%\n")

    print("The mean accuracy of the " + choice + " during training was " + "{:.2f}".format(patients_scores['train_score'].mean() * 100) + "%")

    generate_patients_heatmap(learner, cv)


#model_learning_on_emails()
model_learning_on_patients()