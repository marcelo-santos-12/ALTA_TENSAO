import numpy as np
import matplotlib.pyplot as plt

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

def format_table(table_csv, remove=False):
    # Removendo colunas desnecessarias e organizando dados no formato entrada e saida
    table_csv = table_csv.drop(columns=['ano'], axis=0)
    y_data = table_csv['area_devastada']

    #if remove:
    #    table_csv = table_csv.drop(columns=['taxa_desemprego', 'analfabetos', 'população'], axis=0)


    x_data = table_csv.drop(columns=['area_devastada'], axis=0)
    
    #x_data = x_data[['exportacao', 'homicidios', 'renda_domiciliar_per_capita', 'PIB']]

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

def plot_info(_id, n_cv, y_data, y_pred, mae, mean_mae, std_mae, features_importance, path_results):
    
    folds_range = np.arange(n_cv)
    plt.title(_id)

    # plotting y true
    plt.plot(folds_range, y_data, label='Real', color='blue')
    
    # plotting y pred
    plt.scatter(folds_range, y_pred, label='Predict', color='red')
    plt.xlabel('n-Fold')
    plt.ylabel('Devastated Area')
    plt.legend()
    plt.savefig(path_results + '/' + _id+'.png')
    plt.show()
    plt.close()

    # plotting absolute error
    plt.title('Absolute Error' + ' - ' + _id)
    plt.scatter(folds_range, np.abs(mae), color='red', label='Absolute Error')

    # plotting standard deviation 
    plt.plot(folds_range, mean_mae + 2 * np.repeat(std_mae, repeats=n_cv), 'k--', label='Standard Deviation', color='black')
    plt.plot(folds_range, mean_mae - 2 * np.repeat(std_mae, repeats=n_cv), 'k--', color='black')

    # plotting mean error
    plt.plot(folds_range, np.repeat(mean_mae, repeats=n_cv), 'k--', label='MAE', color='blue')
    plt.xlabel('n-Fold')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.savefig(path_results + '/' + 'error_' + _id + '.png')
    plt.close()

    plt.title('Features Importance - ' + _id)
    plt.xlabel('Importance (%)')
    plt.barh(y=features_importance[1], width=100*features_importance[0], color='blue')
    plt.savefig(path_results + '/' + 'feat_import_' + _id + '.png')
    plt.close()