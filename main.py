import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from utils import *
import os


def main():
    ##################### Aquisition and Data Configuring ########################

    table_csv = pd.read_csv('variaveis_jan_2020_pa.csv')
    
    x_data, y_data = format_table_acc(table_csv)

    features = {
    'rede_rodoviaria': 'RR',
    'rede_rodoviaria_clandestina': 'RDC',
    'população': 'POP',
    'renda_domiciliar_per_capita': 'RDPC',
    'taxa_desemprego': 'TD',
    'homicidios': 'HOM',
    'PIB': 'PIB',
    'exportacao': 'EXP',
    'analfabetos': 'ANA'
    }

    ##################### Model Training ########################
    # Metrics
    # MAE: Mean Absolute Error
    path_results = 'resultados'
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    # Using cross-validation Leave-one-out
    n_cv = x_data.shape[0]

    rf = RandomForestRegressor()
    rf_parameters = {
        'criterion': ['mse', 'mae'],
        'n_estimators': [5, 11, 51, 101],
        'max_depth': [10, 50, 100, 200],
    }

    regress = [['Random Forest', rf, rf_parameters]]

    for _id, clf, parameters in regress:
        np.random.seed(100)
        print('Classificando com {}...'.format(_id))

        # CROSS-VALIDATION
        clf_grid_search = GridSearchCV(clf, param_grid=parameters, scoring='neg_mean_absolute_error', cv=x_data.shape[0],)


        print('Iniciando GridSearch...')
        results_grid_search = clf_grid_search.fit(X=x_data, y=y_data)
        print('Melhor Parametro: {}'.format(results_grid_search.best_params_))
        print('Melhor MAE: '.format(np.round(results_grid_search.best_score_, 2)))
        print(35 * '- ')

        mae = []
        kfolds = k_fold(n_cv)
        y_pred = []

        min_mae = 1000000

        for i, (train, test) in enumerate(kfolds):

            best_clf = clf_grid_search.best_estimator_

            y_test = best_clf.fit(x_data[train], y_data[train]).predict(x_data[test].reshape(1, -1))

            y_true = y_data[test]

            y_pred.append(y_test)

            i_mae = MAE(y_true, y_test[0])

            mae.append(i_mae)

            if i_mae < min_mae:
                min_mae = i_mae
                best_clf_fold = best_clf

        mean_mae = np.mean(mae)
        std_mae = np.std(mae)
        print ('MAE: ', mean_mae)
        print ('STD: ', std_mae)
        print(35 * '* ')

        features_importance = [best_clf.feature_importances_, list(features.values())]
        
        # Plotting Results
        plot_info(_id, n_cv, y_data, y_pred, mae, mean_mae, std_mae, features_importance, path_results)


if __name__ == '__main__':

    main()
