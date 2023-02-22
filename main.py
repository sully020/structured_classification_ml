import pandas as pan
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def create_emails_dframe():
    emails = pan.read_csv('emails.csv')
    x = emails.drop(columns = ['Email No.', 'class_val'])
    y = emails['class_val']
    return x, y
    
def create_patients_dframe():
    patients = pan.read_csv('diabetes.csv')
    x = patients.drop(columns = 'class_val')
    y = patients['class_val']
    return x, y

def decision_tree_emails():
    x, y = create_emails_dframe()
    dtree = DecisionTreeClassifier(criterion = 'gini', max_depth = 25) # ~lg(features) decisions with padding, 
    scores = cross_val_score(dtree, x, y, cv = 10)                                
    return scores

def decision_tree_patients():
    x, y = create_patients_dframe()
    dtree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 6) 
    scores = cross_val_score(dtree, x, y)                                                   
    return scores


def neural_network_emails():
    x, y = create_emails_dframe()
    neural_net = MLPClassifier()
    scores = cross_val_score(neural_net, x, y)                                                   
    return scores

def neural_network_patients():
    x, y = create_patients_dframe()
    neural_net = MLPClassifier()
    scores = cross_val_score(neural_net, x, y)                                                   
    return scores


def kNN_emails():
    x, y = create_emails_dframe() 
    kNN = KNeighborsClassifier()
    scores = cross_val_score(kNN, x, y)                                                    
    return scores

def kNN_patients():
    x, y = create_patients_dframe()
    kNN = KNeighborsClassifier()
    scores = cross_val_score(kNN, x, y)                                                    
    return scores


def random_forest_emails():
    x, y = create_emails_dframe() 
    random_forest = RandomForestClassifier()
    scores = cross_val_score(random_forest, x, y)                                                    
    return scores

def random_forest_patients():
    x, y = create_patients_dframe() 
    random_forest = RandomForestClassifier()
    scores = cross_val_score(random_forest, x, y)                                                     
    return scores


def test_decision_trees():
    email_scores = decision_tree_emails()
    patient_scores = decision_tree_patients()
    print("The mean accuracy of the Decision Tree used to classify spam e-mails by \
10-fold cross-validation is: " + "{:.1f}".format(email_scores.mean() * 100) + "%")
    print("The mean accuracy of the Decision Tree used to classify diabetes by \
5-fold cross-validation is: " + "{:.1f}".format(patient_scores.mean() * 100) + "%")

def test_neural_networks():
    email_scores = neural_network_emails()
    patient_scores = neural_network_patients()
    print("The mean accuracy of the Neural Network used to classify spam e-mails by \
5-fold cross-validation is: " + "{:.1f}".format(email_scores.mean() * 100) + "%")
    print("The mean accuracy of the Neural Network used to classify diabetes by \
5-fold cross-validation is: " + "{:.1f}".format(patient_scores.mean() * 100) + "%")

def test_kNN():
    email_scores = kNN_emails()
    patient_scores = kNN_patients()
    print("The mean accuracy of the k-Nearest Neighbor algorithm used to classify spam e-mails by \
5-fold cross-validation is: " + "{:.1f}".format(email_scores.mean() * 100) + "%")
    print("The mean accuracy of the k-Nearest Neighbor algorithm used to classify diabetes by \
5-fold cross-validation is: " + "{:.1f}".format(patient_scores.mean() * 100) + "%")

def test_random_forest():
    email_scores = random_forest_emails()
    patient_scores = random_forest_patients()
    print("The mean accuracy of the Random Forest used to classify spam e-mails by \
5-fold cross-validation is: " + "{:.1f}".format(email_scores.mean() * 100) + "%")
    print("The mean accuracy of the Random Forest used to classify diabetes by \
5-fold cross-validation is: " + "{:.1f}".format(patient_scores.mean() * 100) + "%")


test_decision_trees()
#test_neural_networks()
#test_kNN()
#test_random_forest()