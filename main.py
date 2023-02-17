import pandas as pan
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def create_emails_df():
    emails = pan.read_csv('emails.csv')
    x = emails.drop(columns = ['Email No.', 'class_val'])
    y = emails['class_val']
    return x, y
    #print(x.head(5))

def create_patient_df():
    patients = pan.read_csv('diabetes.csv')
    x = patients.drop(columns = 'class_val')
    y = patients['class_val']
    return x, y
    #print(x.head(5))

def separate_data():
    x, y = create_emails_df
    cross_val_score(X=x, y=y)

def decision_tree_emails():
    dtree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 20, max_features = "sqrt") # ~lg(features) decisions with padding, 
    dtree.fit(X=x, y=y)                                                                                     # semi-aggressive pruning(?)

