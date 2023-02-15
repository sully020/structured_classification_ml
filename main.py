import pandas as pan
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def classify_emails():

    emails = pan.read_csv('emails.csv')
    dtree = DecisionTreeClassifier(criterion = 'entropy')
    print(emails[0])

classify_emails()
