import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, recall_score, accuracy_score, precision_score, f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from utils import *


def main():
    ##################### Aquisicao e Tratamento de dados ########################

    table_csv = pd.read_csv('variaveis_jan_2020_pa.csv')

    x_data, y_data = format_table(table_csv)

    ##################### Treinamento do Modelo ########################
    # Metricas para Regressao:
    # MSE: Erro medio quadratico
    # MAE: Erro medio absoluto

    # Usando Cross VAlidation com K igual ao numero de amostras
    n_cv = x_data.shape[0]

    clf = SVR()
    clf = KNeighborsClassifier()
    clf = RandomForestClassifier()
    mae = []

    kfolds = k_fold(n_cv)

    for i, (train, test) in enumerate(kfolds):

        y_test = clf.fit(x_data[train], y_data[train]).predict(x_data[test].reshape(1, -1))
        y_true = y_data[test]
        
        i_mae = MAE(y_true, y_test[0])
        
        mae.append(i_mae)
    
    print(np.mean(np.array(mae)))
    print(np.std(np.array(mae)))
    quit()

    svm = SVR()
    mlp = MLPClassifier()
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier()
    knn = KNeighborsClassifier()
    
    svm_parameters = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [1, 10, 100],
        'gamma': [0.0001, 0.00001, 0.000001]
    }
    mlp_parameters = {
        'hidden_layer_sizes': [(5,), (10,), (20,), (10, 10)],
        'solver': ['adam', 'sgd'],
        'activation': ['relu', 'identity', 'logistic', 'tanh'],
        'max_iter': [50, 100, 200]
    }
    dt_parameters = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [3, 5, 10, 50],
    }
    rf_parameters = {
        'n_estimators': [5, 11, 51, 101],
        'criterion': ['gini', 'entropy'],
        'max_depth': [10, 50, 100, 200],
    }
    knn_parameters = {
        'n_neighbors': [1, 5, 9],
        'weights' : ['uniform', 'distance'],
        'algorithm': ['kd_tree', 'ball_tree'],
        'p': [1, 2] # Manhatan and Euclidian distance, respectivity
    }

    classifiers = [['SVM', svm, svm_parameters], ['MLP', mlp, mlp_parameters], \
                ['Decision Trees', dt, dt_parameters], ['Random Forest', rf, rf_parameters], \
                ['K-Nearest Neighbor', knn, knn_parameters]]
    
    classifiers = [['MLP', mlp, mlp_parameters]] #, ['Decision Trees', dt, dt_parameters], ['Random Forest', rf, rf_parameters], \
                # ['K-Nearest Neighbor', knn, knn_parameters]]
    
    # METRICAS A SEREM ANALISADAS
    mae = []
    mse = []
    for _id, clf, parameters in classifiers:
        np.random.seed(10)
        cv = StratifiedKFold(n_splits=n_cv)
        print(35 * ' * ')
        print('Classificando com {}...'.format(_id))
        
        # CROSS-VALIDATION
        clf_grid_search = GridSearchCV(clf, param_grid=parameters, scoring='accuracy', cv=n_cv)

        print('Iniciando GridSearch...')
        results_grid_search = clf_grid_search.fit(X=x_data, y=y_data)

        print('Melhor Parametro: {}'.format(results_grid_search.best_params_))
        print('Melhor F1Score: '.format(np.round(results_grid_search.best_score_, 2)))
        print(35 * '- ')
        print()

        np.random.seed(10)

        for i, (train, test) in enumerate(cv.split(x_data, y_data)):

            best_clf = clf_grid_search.best_estimator_
            
            probas_ = best_clf.fit(x_data[train], y_data[train]).predict_proba(x_data[test])
            
            # COMPUTANDO METRICAS PARA CADA FOLD
            i_mse = recall_score(y_data[test], np.round(probas_[:, 1]))
            i_mae = precision_score(y_data[test], np.round(probas_[:, 1]))
            
            mse.append(i_mse)
            mae.append(i_mae)
        
        # PRINTANDO RESULTADOS GERAIS DO MODELO
        results = np.asarray([mse, mae])
        id_results = ['MSE', 'MAE']
        
        print('Resultados do Classificador: {}'.format(_id))
        for i, res in enumerate(results):
            results_mean = 100 * np.round(res.mean(), 4)
            results_std = 100 * np.round(res.std(), 4)
            score = id_results[i]
            print('Resultados {}: {}'.format(score, res))
            print('Média: {}%'.format(results_mean))
            print('Desvio Padrão: {}%'.format(results_std))
            print(35 * '- ')

if __name__ == '__main__':

    main()
