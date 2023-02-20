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

def create_patients_df():
    patients = pan.read_csv('diabetes.csv')
    x = patients.drop(columns = 'class_val')
    y = patients['class_val']
    return x, y
    #print(x.head(5))

def decision_tree_emails():
    x, y = create_emails_df()
    dtree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 20, max_features = "sqrt") # ~lg(features) decisions with padding, 
    scores = cross_val_score(dtree, X=x, y=y)                                                    # semi-aggressive pruning(?)
    print(scores)

def decision_tree_patients():
    x, y = create_patients_df()
    dtree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 20, max_features = "sqrt") # ~lg(features) decisions with padding, 
    scores = cross_val_score(dtree, X=x, y=y)                                                    # semi-aggressive pruning(?)
    print(scores)

def test_decision_trees():
    decision_tree_emails()
    decision_tree_patients()