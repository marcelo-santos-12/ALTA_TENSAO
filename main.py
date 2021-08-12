import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from utils.utils import (format_table, format_table_acc, k_fold, MAE, plot_results)
from utils.sensibility_analisys import sensibility_analysis
import os


features = {
    'rede_rodoviaria': 'RR',
    'rede_rodoviaria_clandestina': 'RRC',
    'população': 'POP',
    'renda_domiciliar_per_capita': 'RDPC',
    'taxa_desemprego': 'TD',
    'homicidios': 'HOM',
    'PIB': 'PIB',
    'exportacao': 'EXP',
    'analfabetos': 'ANA'
    }


def main():
    DEBUG = False

    ##################### Aquisition and Data Configuring ########################
    var = 'jan_novo'
    table_csv = pd.read_csv('dados/variaveis_jan_novo_2020_pa.csv')
    x_data, y_data = format_table_acc(table_csv)

    var = 'julho'
    table_csv = pd.read_csv('variaveis_julho_2021_pa.csv')
    x_data, y_data = format_table(table_csv)

    ######### Select samples for sensibility analysis ###########
    years_sens = ['1990', '2003','2016']
    all_years = np.arange(1988, 2019)
    ind = np.arange(len(all_years))
    years_str = [str(value) for value in all_years]
    arr_index = list(zip(ind, years_str))

    #P = .1
    ##################### Model Training ########################
    # Metrics
    # MAE: Mean Absolute Error

    path_results = 'resultados_' + var
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    ############## Using cross-validation Leave-one-out #########
    n_cv = x_data.shape[0]

    ############# Regressors and its parameters
    rf  = RandomForestRegressor()
    dt  = DecisionTreeRegressor()
    knn = KNeighborsRegressor()

    rf_parameters = {
        'criterion': ['mse', 'mae'],
        'n_estimators': [5, 11, 51, 101],
        'max_depth': [10, 50, 100, 200]
    }

    dt_parameters = {
        'criterion': ['mse', 'mae'],
        'splitter':  ['best', 'random'],
        'max_depth': [3, 5, 10, 50]
    }
    
    knn_parameters = {
        'n_neighbors': [1, 5, 9],
        'weights': ['uniform', 'distance'],
        'algorithm': ['kd_tree', 'ball_tree'],
        'p': [0, 1]
    }

    regress = [
        ['Random Forest', rf, rf_parameters],
        #['Decision Trees', dt, dt_parameters],
        #['K Nearest Neighbor', knn, knn_parameters]
    ]

    for P in [.1, .15, .2, .25]:
        df_sensibility_rn = pd.DataFrame(columns=['RN', 'DA TRUE', 'DA PREDICT', 'RN VAR', 'DA VAR', 'EFFECT'])
    
        df_sensibility_crn = pd.DataFrame(columns=['CRN', 'DA TRUE', 'DA PREDICT', 'CRN VAR', 'DA VAR', 'EFFECT'])
    
        for _id, clf, parameters in regress:
            np.random.seed(100) # to reprodutibility
            print('Classificando com {}...'.format(_id))
            print('Percent: ', P)

            # CROSS-VALIDATION LEAVE-ONE-OUT
            clf_grid_search = GridSearchCV(clf, param_grid=parameters, scoring='neg_mean_absolute_error', cv=n_cv,)
            
            print(35 * '- ')
            print('Iniciando GridSearch...')
            results_grid_search = clf_grid_search.fit(X=x_data, y=y_data)
            print('Melhor Parametro: {}'.format(results_grid_search.best_params_))
            print('Melhor Erro Medio Absoluto: ', -float(np.round(results_grid_search.best_score_, 2)))
            print(35 * '- ')

            mae = []
            kfolds = k_fold(n_cv)

            y_pred = []

            for i, (train, test) in enumerate(kfolds):

                best_clf = results_grid_search.best_estimator_

                best_clf.fit(x_data[train], y_data[train])
                y_test = best_clf.predict(x_data[test].reshape(1, -1))

                ##########################################################
                ## SENSITIVITY ANALISYS
                # Variable Index | 0 --> RN, 1 --> CRN
                if i in ind:
                    # CRN
                    df_sensibility_crn = sensibility_analysis(x_data[test],
                                                y_data[test],
                                                i=1, p=P,
                                                model=best_clf,
                                                variable='CRN',
                                                df_sens=df_sensibility_crn)

                    # RN
                    df_sensibility_rn = sensibility_analysis(x_data[test],
                                                y_data[test],
                                                i=0, p=P,
                                                model=best_clf,
                                                variable='RN',
                                                df_sens=df_sensibility_rn)

                y_true = y_data[test]

                y_pred.append(y_test)

                i_mae = MAE(y_true, y_test[0])

                mae.append(i_mae)

                if i == 0:
                    min_mae = i_mae
                    continue

                if i_mae < min_mae:
                    if DEBUG:
                        print('Indice do Melhor CLF:', i)
                    best_clf_fold = best_clf
                    min_mae = i_mae

            mean_mae = np.mean(mae)
            std_mae = np.std(mae)

            if DEBUG:
                print ('MAE: ', mean_mae)
                print ('STD: ', std_mae)
                print('Minimum MAE: ', min_mae)
                print(35 * '* ')

            features_importance = [best_clf_fold.feature_importances_, list(features.values())]
            for value, key in zip(*features_importance):
                print(key, 100*np.round(value, 2), '%')

            df_sensibility_crn.to_excel(path_results+'/sensibility_crn_' + str(P) + '.xlsx')

            df_sensibility_rn.to_excel(path_results+'/sensibility_rn_' + str(P) + '.xlsx')

            # Plotting Results
            plot_results(_id, n_cv, y_data, y_pred, mae, mean_mae, std_mae, features_importance, path_results, arr_index, years_sens)


if __name__ == '__main__':

    main()
