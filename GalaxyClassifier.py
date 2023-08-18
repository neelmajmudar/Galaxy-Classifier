from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

data = pd.read_csv('Galaxydata.csv')

#print(data.columns)
input_data = data[['P_EL', 'P_CW', 'P_ACW', 'P_EDGE','P_DK', 'P_MG', 'P_CS', 'P_EL_DEBIASED', 'P_CS_DEBIASED']]
labels = data[['SPIRAL', 'ELLIPTICAL', 'UNCERTAIN']]
all_data = pd.concat([input_data, labels], axis=1)


#Histogram for Features/Class 
def plot_histogram():
    for feature in all_data.columns:
        plt.figure(figsize=(8,5))
        for classlabel in labels:
            sns.histplot(data=all_data, x=feature, hue=classlabel, multiple='stack', bins=20)
            plt.title(f'Histogram: {feature} Distribution by Class')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.legend(title=str(classlabel))
            plt.show() 
#plot_histogram()

#Data Correlation
def plot_heatmap():
    correlation = all_data.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.show()
#plot_heatmap()

output_accuracies = [0.9296798389088922, 0.9688746827957392, 0.8985844642897245]
output_accuracies = [i * 100 for i in output_accuracies]

#Accuracies per Class
def plot_accuracies(accuracies):
    class_labels = ['SPIRAL', 'ELLIPTICAL', 'UNCERTAIN']
    plt.figure(figsize=(8,6))
    plt.bar(class_labels, accuracies, color=['blue', 'red', 'yellow'])
    plt.yscale('log')
    plt.title('Accuracies for Each Class')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.show()
#plot_accuracies(output_accuracies)

#Feature Importances
def plot_importances(importances):
    plt.figure(figsize=(10, 6))
    plt.bar(data.columns[4:13], importances)
    plt.title('Feature Importances')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45, ha='right')
    plt.show()

#Confusion Matrix
def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',xticklabels=labels.columns, yticklabels=labels.columns)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

param_grid = {
    'n_estimators': [50, 100, 150],    # Number of trees in the forest
    'max_depth': [None, 10, 20],       # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],   # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]      # Minimum number of samples required to be at a leaf node
}


#Hyper Parameter Tuning
def hypertuning(param_grid, x_train, y_train, x_test, y_test):
    random_forest = RandomForestClassifier(random_state= 42)
    grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_search.fit(x_train, y_train)
    best_random_forest = grid_search.best_estimator_
    best_prediction = best_random_forest.predict(x_test)
    best_accuracy = accuracy_score(y_test, best_prediction)
    result = {"Best Parameters:": grid_search.best_params_, "Best Model Accuracy:":best_accuracy}
    return result

#Best Parameters: {'max_depth': 20, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 100}
#Best model accuracy: 0.9033004214418852
#Runtime on this computer = [Finished in 4770.3s] 

best_params = {'max_depth': 20, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 100}

#Standard Model: Accuracy = 0.9
scaler = StandardScaler()

x_train, x_test, y_train, y_test = train_test_split(input_data, labels, test_size = 0.2, random_state = 42)

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)

random_forest = RandomForestClassifier(random_state= 42, **best_params)

random_forest.fit(x_train, y_train)

prediction = random_forest.predict(x_test)
#Plotting Feature Importances
importances = random_forest.feature_importances_
#Plotting Confusion Matrix
conf_matrix = confusion_matrix(y_test.values.argmax(axis=1), prediction.argmax(axis=1))
#Plotting Accuracies
accuracies = []
for i in range(3):
    accuracy = accuracy_score(y_test.iloc[:, i], prediction[:, i])
    accuracies.append(accuracy)
model_accuracy = accuracy_score(y_test, prediction)

#plot_confusion_matrix(conf_matrix)
#plot_importances(importances)
