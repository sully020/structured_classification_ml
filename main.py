import pandas as pan
import seaborn as sea
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
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
    cv = 5
    match choice:
        case 'decision tree':
            learner = DecisionTreeClassifier(max_depth = 24) # No manual pruning worked
            cv = 10
        case 'neural network':
            learner = MLPClassifier()
        case 'k neighbors':
            learner = KNeighborsClassifier()
        case 'random forest':
            learner = RandomForestClassifier()
        case _:
            raise ValueError("Inappropriate argument value of " + str(choice) + " entered.")

    return learner, cv

def score_email_predictions(learner, cv):
    x, y = create_emails_dframe()
    scores = cross_val_score(learner, x, y, cv = cv)                      
    return scores

def generate_email_heatmap(learner, cv):
    x, y = create_emails_dframe()
    preds = cross_val_predict(learner, x, y, cv = cv) 
    cf_matrix = confusion_matrix(y, preds)    
    sea.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Reds')
    plt.show()


def model_learning_on_emails():
    choice = input("Please enter your choice of learning model for the spam e-mails dataset: \
    1.) 'decision tree' 2.) 'neural network' 3.) 'k neighbors' 4.) 'random forest': ")
    learner, cv = choose_email_learner(choice)
    email_scores = score_email_predictions(learner, cv)

    print("The mean accuracy of the " + choice + " used to classify spam e-mails by " 
    + str(cv) + "-fold cross-validation is: " + "{:.2f}".format(email_scores.mean() * 100) + "%") #CHANGE TO .0f after analysis

    generate_email_heatmap(learner, cv)



def create_patients_dframe():
    patients = pan.read_csv('diabetes.csv')
    x = patients.drop(columns = 'class_val')
    y = patients['class_val']
    return x, y

def choose_patients_learner(choice):
    cv = 5
    match choice:
        case 'decision tree':
            learner = DecisionTreeClassifier() # No manual pruning worked
        case 'neural network':
            learner = MLPClassifier()
        case 'k neighbors':
            learner = KNeighborsClassifier()
        case 'random forest':
            learner = RandomForestClassifier()
        case _:
            raise ValueError("Inappropriate argument value of " + str(choice) + " entered.")

    return learner, cv

def score_patients_predictions(learner, cv):
    x, y = create_patients_dframe()
    scores = cross_val_score(learner, x, y, cv = cv)                      
    return scores

def generate_patients_heatmap(learner, cv):
    x, y = create_patients_dframe()
    preds = cross_val_predict(learner, x, y, cv = cv) 
    cf_matrix = confusion_matrix(y, preds)    
    sea.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Reds')
    plt.show()

def model_learning_on_patients():
    choice = input("Please enter your choice of learning model for the diabetes patients dataset: \
    1.) 'decision tree' 2.) 'neural network' 3.) 'k neighbors' 4.) 'random forest': ")
    learner, cv = choose_patients_learner(choice)
    patients_scores = score_patients_predictions(learner, cv)
    print("The mean accuracy of the Decision Tree used to classify diabetes by " \
    + str(cv) + " fold cross-validation is: " + "{:.2f}".format(patients_scores.mean() * 100) + "%")


# Leave us alone
def decision_tree_patients():
    x, y = create_patients_dframe()
    dtree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 6) 
    scores = cross_val_score(dtree, x, y)                                                   
    return scores

def neural_network_patients():
    x, y = create_patients_dframe()
    neural_net = MLPClassifier()
    scores = cross_val_score(neural_net, x, y)                                                   
    return scores

def kNN_patients():
    x, y = create_patients_dframe()
    kNN = KNeighborsClassifier()
    scores = cross_val_score(kNN, x, y)                                                    
    return scores

def random_forest_patients():
    x, y = create_patients_dframe() 
    random_forest = RandomForestClassifier()
    scores = cross_val_score(random_forest, x, y)                                                     
    return scores

test_emails_r()
test_patients_r()