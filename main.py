import pandas as pan
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def decisiontree_emails():
    # This will be a separate atomized function
    emails = pan.read_csv('emails.csv')
    dtree = DecisionTreeClassifier(criterion = 'entropy')

    x = emails.drop(columns = ['Email No.', 'class_val'])
    y = emails['class_val']
    print(x.head(5))

def decisiontree_patients():
    patients = pan.read_csv('diabetes.csv')
    x = patients.drop(columns = 'class_val')
    y = patients['class_val']
    print(x.head(5))

decisiontree_emails()
decisiontree_patients()

