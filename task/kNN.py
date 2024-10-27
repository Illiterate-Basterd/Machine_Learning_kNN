from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Распределение признаков классов на параллельных координатах------------------------------------------------

def DataView(data: pd.DataFrame):
    plt.figure(figsize = (15, 10))
    pd.plotting.parallel_coordinates(data, 'label')
    plt.title('Parallel Coordinates Plot', fontsize = 20, fontweight = 'bold')
    plt.xlabel('Features', fontsize = 15)
    plt.ylabel('Features values', fontsize = 15)
    plt.legend(loc = 1, prop = {'size': 15}, frameon = True, shadow = True, facecolor = 'white', edgecolor = 'black')
    plt.show()

# Подготовка и обработка данных------------------------------------------------------------------------------

def DataPrep():
    iris = load_iris()
    iris_data = iris.data
    iris_labels = iris.target

    labels = np.reshape(iris_labels, (150, 1))
    final_iris_data = np.concatenate([iris_data, labels], axis=1)

    iris_dataset = pd.DataFrame(final_iris_data)
    features = iris.feature_names
    feature_labels = np.append(features, 'label')
    iris_dataset.columns = feature_labels

    iris_dataset['label'] = iris_dataset['label'].replace(0, 'Setosa')
    iris_dataset['label'] = iris_dataset['label'].replace(1, 'Versicolor')
    iris_dataset['label'] = iris_dataset['label'].replace(2, 'Virginica')

    print(iris_dataset)

    X = iris_dataset[features].values
    y = iris_dataset['label'].values

    y = LabelEncoder().fit_transform(y)

    return (iris_dataset, features, X, y)

# Обучение и применение классификатора-------------------------------------------------------------------------

def kNNLearn():
    classifier = KNeighborsClassifier(n_neighbors = 11)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of the model is', round(accuracy, 4))

# Кросс-валидация для регулировки параметров модели------------------------------------------------------------

def CrossVal():
    k_list = list(range(1, 50, 2))
    cv_scores = []

    for k in k_list:
        knn = KNeighborsClassifier(n_neighbors = k)
        scores = cross_val_score(knn, X_train, y_train, cv = 10, scoring = 'accuracy')
        cv_scores.append(scores.mean())
    
    MSE = [1 - x for x in cv_scores]
    plt.figure(figsize = (15, 10))
    plt.title('The optimal number of neighbors', fontsize = 20, fontweight = 'bold')
    plt.xlabel('Number of Neighbors K', fontsize = 15)
    plt.ylabel('Misclassification Error', fontsize = 15)
    sns.set_style('whitegrid')
    plt.plot(k_list, MSE)
    plt.grid()
    plt.show()
    
    best_k = k_list[MSE.index(min(MSE))]
    print('The optimal number of neighbors is', best_k)

# Уменьшение размерности данных (Принцип главных компонент)------------------------------------------------------

def PCAFunc():
    x = iris_dataset.loc[:, features].values
    x = StandardScaler().fit_transform(x)

    pca_iris = PCA(n_components=2)
    principalComponents_iris = pca_iris.fit_transform(x)

    principal_iris_DF = pd.DataFrame(data = principalComponents_iris, columns = ['principal component 1', 'principal component 2'])

    print(principal_iris_DF.tail())

    print('Explained variation per principal component: {}'.format(pca_iris.explained_variance_ratio_))

    plt.figure(figsize=(8, 7))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component - 1', fontsize=20)
    plt.ylabel('Principal Component - 2', fontsize=20)
    plt.title('Principal Component Analysis of Iris Dataset', fontsize=20)
    targets = ['Setosa', 'Versicolor', 'Virginica']
    colors = ['g', 'r', 'blue']

    for target, color in zip(targets, colors):
        indicesToKeep = iris_dataset['label'] == target
        plt.scatter(principal_iris_DF.loc[indicesToKeep, 'principal component 1'], principal_iris_DF.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

    plt.legend(targets, prop = {'size': 15})
    plt.show()

#-------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    iris_dataset, features, X, y = DataPrep()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 103)
    
    DataView(iris_dataset)

    kNNLearn()

    CrossVal()
    
    PCAFunc()