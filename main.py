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
    dtree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 20, max_features = "sqrt") # ~lg(features) decisions with padding, 
    scores = cross_val_score(dtree, x, y)                                                        # semi-aggressive pruning(?)
    print(scores)

def decision_tree_patients():
    x, y = create_patients_dframe()
    dtree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 20, max_features = "sqrt") 
    scores = cross_val_score(dtree, x, y)                                                   
    print(scores)


def neural_network_emails():
    x, y = create_emails_dframe()
    neural_net = MLPClassifier(6, 'logistic', solver = 'sgd', learning_rate = 'adaptive', max_iter = 450)
    scores = cross_val_score(neural_net, x, y)                                                   
    print(scores)

def neural_network_patients():
    x, y = create_patients_dframe()
    neural_net = MLPClassifier(6, 'logistic', solver = 'sgd', learning_rate = 'constant', max_iter = 450)
    scores = cross_val_score(neural_net, x, y)                                                   
    print(scores)


def kNN_emails():
    x, y = create_emails_dframe() 
    kNN = KNeighborsClassifier(500)
    scores = cross_val_score(kNN, x, y)                                                    
    print(scores)

def kNN_patients():
    x, y = create_patients_dframe()
    kNN = KNeighborsClassifier()
    scores = cross_val_score(kNN, x, y)                                                    
    print(scores)


def random_forest_emails():
    x, y = create_emails_dframe() 
    random_forest = RandomForestClassifier()
    scores = cross_val_score(random_forest, x, y)                                                    
    print(scores)

def random_forest_patients():
    x, y = create_emails_dframe() 
    random_forest = RandomForestClassifier()
    scores = cross_val_score(random_forest, x, y)                                                     
    print(scores)


def test_decision_trees():
    decision_tree_emails()
    decision_tree_patients()

def test_neural_networks():
    neural_network_emails()
    neural_network_patients()

def test_kNN():
    kNN_emails()
    kNN_patients()

def test_random_forest():
    random_forest_emails()
    random_forest_patients()


test_decision_trees()
#test_neural_networks()
#test_kNN()
#test_random_forest()