import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from utils import *


def main():
    ##################### Aquisicao e Tratamento de dados ########################

    table_csv = pd.read_csv('variaveis_jan_2020_pa.csv')
    #table_csv = pd.read_csv('interpolation_without_normalization_jan_20.csv')

    x_data, y_data = format_table(table_csv, remove=False)

    ##################### Treinamento do Modelo ########################
    # Metricas
    # MSE: Erro medio quadratico
    
    # Usando Cross VAlidation com K igual ao numero de amostras
    n_cv = x_data.shape[0]

    svm = SVR()
    mlp = MLPRegressor()
    dt = DecisionTreeRegressor()
    rf = RandomForestRegressor()
    knn = KNeighborsRegressor()
    
    svm_parameters = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [1, 10, 100],
        'gamma': [0.0001, 0.00001, 0.000001]
    }
    mlp_parameters = {
        'hidden_layer_sizes': [(5,), (10,), (20,), (10, 10)],
        'solver': ['adam', 'sgd'],
        'activation': ['relu', 'identity', 'logistic', 'tanh'],
        'max_iter': [500]
    }
    dt_parameters = {
        'criterion': ['mse', 'mae'],
        'splitter': ['best', 'random'],
        'max_depth': [3, 5, 10, 50],
    }
    rf_parameters = {
        'criterion': ['mse', 'mae'],
        'n_estimators': [5, 11, 51, 101],
        'max_depth': [10, 50, 100, 200],
    }
    knn_parameters = {
        'n_neighbors': [1, 5, 9],
        'weights' : ['uniform', 'distance'],
        'algorithm': ['kd_tree', 'ball_tree'],
        'p': [1, 2] # Manhatan and Euclidian distance, respectivity
    }

    #regress = [['Decision Trees', dt, dt_parameters], ['Random Forest', rf, rf_parameters], \
    #            ['K-Nearest Neighbor', knn, knn_parameters], ['SVM', svm, svm_parameters], ['MLP', mlp, mlp_parameters],]
    
    regress = [['Decision Trees', dt, dt_parameters], ['Random Forest', rf, rf_parameters], \
                ['K-Nearest Neighbor', knn, knn_parameters]]

    for _id, clf, parameters in regress:
        np.random.seed(100)
        print('Classificando com {}...'.format(_id))
        
        # CROSS-VALIDATION
        clf_grid_search = GridSearchCV(clf, param_grid=parameters, scoring='neg_mean_absolute_error', cv=31,)

        print('Iniciando GridSearch...')
        results_grid_search = clf_grid_search.fit(X=x_data, y=y_data)
        print('Melhor Parametro: {}'.format(results_grid_search.best_params_))
        print('Melhor MSE: '.format(np.round(results_grid_search.best_score_, 2)))
        print(35 * '- ')
        print()
        
        mae = []
        kfolds = k_fold(n_cv)
        y_pred = []

        for i, (train, test) in enumerate(kfolds):
            
            best_clf = clf_grid_search.best_estimator_
            
            y_test = best_clf.fit(x_data[train], y_data[train]).predict(x_data[test].reshape(1, -1))

            y_true = y_data[test]
            
            y_pred.append(y_test)
            
            i_mae = MAE(y_true, y_test[0])
            
            mae.append(i_mae)

        mean_mae = np.mean(mae)
        std_mae = np.std(mae)
        
        print ('MAE: ', mean_mae)
        print ('STD: ', std_mae)

        print(35 * '* ')
        print()

        folds_range = np.arange(n_cv)

        plt.title(_id)
        # plotting y true
        plt.plot(folds_range, y_data, label='Valores Reais', color='blue')
        
        # plotting y pred
        plt.scatter(folds_range, y_pred, label='Valores Preditos', color='red')
        plt.legend()
        plt.savefig(_id+'.png')
        plt.close()


        plt.title('Absolute Error' + ' - ' + _id)
        # plotting standard deviation 
        plt.plot(folds_range, mean_mae + 2 * np.repeat(std_mae, repeats=n_cv), 'k--', label='Standard Deviation', color='black')
        plt.plot(folds_range, mean_mae - 2 * np.repeat(std_mae, repeats=n_cv), 'k--', color='black')
        # plotting absolute error
        plt.scatter(folds_range, np.abs(mae), color='red', label='Absolute Error')
        
        # plotting mean error
        plt.plot(folds_range, np.repeat(mean_mae, repeats=n_cv), 'k--', label='MSE', color='blue')
        
        plt.legend()
        plt.savefig('error_' + _id + '.png')
        plt.close()

if __name__ == '__main__':

    main()
