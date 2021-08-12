import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size

def k_fold(cv):
    '''
    Leave-one-out Method.
    KOHAVI, R. A study of cross-validation and bootstrap for accuracy estimation and model selection. In: International joint Conference on artificial intelligence. [S.l.: s.n.], 1995.
    '''
    kfold_train, kfold_test = [], []
    for i in np.arange(cv):
        test = np.arange(cv)
        test = np.delete(test, i)
        kfold_train.append(list(test))
        kfold_test.append(i)
    return zip(kfold_train, kfold_test)

def format_table(table_csv,):
    # Removendo colunas desnecessarias e organizando dados no formato entrada e saida
    table_csv = table_csv.drop(columns=['ano'], axis=0)
    y_data = table_csv['area_devastada']

    x_data = table_csv.drop(columns=['area_devastada'], axis=0)
    
    # Formatando os dados para o formato numerico
    def replace_virg(data):
        if isinstance(data, str):
            return data.replace('.', '').replace(',', '.')
        return data

    for col in x_data.columns:

        values = list(map(replace_virg, x_data[col]))

        x_data[col] = np.asarray(values, dtype=np.double)

    return np.asarray(x_data), np.asarray(y_data)

def format_table_acc(table_csv):
    # Removendo colunas desnecessarias e organizando dados no formato entrada e saida
    table_csv = table_csv.drop(columns=['ano'], axis=0)
    
    y_data = table_csv['area_devastada']
    x_data = table_csv.drop(columns=['area_devastada'], axis=0)

    x_0 = 267393.0285 - y_data.sum()
    
    y_data_new = []
    for i in y_data:
        x_0 += i
        y_data_new.append(x_0)

    y_data = np.asarray(y_data_new)

    # Formatando os dados para o formato numerico
    def replace_virg(data):
        if isinstance(data, str):
            return data.replace('.', '').replace(',', '.')
        return data

    for col in x_data.columns:

        values = list(map(replace_virg, x_data[col]))

        x_data[col] = np.asarray(values, dtype=np.double)

    return np.asarray(x_data), np.asarray(y_data)

def MAE(y_true, y_test):
    return abs(y_true - y_test)

def plot_results(_id, n_cv, y_data, y_pred, mae, mean_mae, std_mae, features_importance, path_results, arr_index, years_sens):
    '''
    Plot all the results necessary and storage on local filesystem.
        Parameters
        ---------
            _id: (str)
                Identifier string for regressor.
            
            n_cv: (int)
                Number of folds used for cross-validatiob Leave-One-Out.
            
            y_data: (array --> 1xN)
                Y true (desired).
            
            y_pred: (array --> 1xN)
                Y predict (given by regressor model).

            mae: (float)
                Mean Absolute Error.

            mean_std: (float)
                Average of MAE.

            std_mae: (float)
                Standard deviation of MAE.

            features_importance: (array --> 1xN)
                Array that contains the importance of each features/variable from dataset for the model.

            path_results: (str)
                Path where the results will be storaged.

            arr_index: (list of list --> [[year (int), year (string)]])
                List that contains the years of the samples from dataset.
                Obs.: used for sensibility analysis.

            years_sens: (tuple or list of string)
                List that contains the years used for sensibility analysis.
        Return
        ------
            None:
                Storage the graphics generated.
    '''

    plt.style.use("ggplot")

    folds_range = np.arange(n_cv)
    plt.title(_id)

    # plotting y true
    plt.plot(folds_range, y_data, label='Real', color='blue')

    y_pred = np.asarray(y_pred).reshape(-1,)

    # plotting y pred
    plt.scatter(folds_range, y_pred, label='Predito', color='red')

    # extract the index from arr_index
    for ind, year in arr_index:

        if year in years_sens:

            plt.plot(ind, y_pred[ind], marker='o', fillstyle='none', color='black', markersize=15, linestyle='none')

            plt.text(
                x=ind,
                y=y_pred[ind] *.93,
                s=year,
                size=8,
            )

    
    plt.xlabel('n-Fold')
    plt.ylabel('Área Devastada (Km²)', fontsize=16)
    plt.legend()
    plt.savefig(path_results + '/' + _id+'.png')
    plt.show()
    plt.close()

    # plotting absolute error
    plt.title('Erro Absoluto' + ' - ' + _id)
    plt.scatter(folds_range, np.abs(mae), color='red', label='Erro Absoluto (Km²)')

    # plotting standard deviation 
    plt.plot(folds_range, mean_mae + 2 * np.repeat(std_mae, repeats=n_cv), 'k--', label='Desvio Padrão (Km²)', color='black')
    plt.plot(folds_range, mean_mae - 2 * np.repeat(std_mae, repeats=n_cv), 'k--', color='black')

    # plotting mean error
    plt.plot(folds_range, np.repeat(mean_mae, repeats=n_cv), 'k--', label='MAE', color='blue')
    plt.xlabel('n-Fold')
    plt.ylabel('Erro Absoluto (Km²)')
    plt.legend()
    plt.savefig(path_results + '/' + 'error_' + _id + '.png')
    plt.close()

    plt.title('Importância dos Atributos - ' + _id)
    plt.xlabel('Importância (%)')
    plt.barh(y=features_importance[1], width=100*features_importance[0], color='blue')
    plt.savefig(path_results + '/' + 'feat_import_' + _id + '.png')
    plt.close()