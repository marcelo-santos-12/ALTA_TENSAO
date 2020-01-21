import numpy as np


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

def format_table(table_csv):
    # Removendo colunas desnecessarias e organizando dados no formato entrada e saida
    x_data = table_csv[['Rede Rodoviária', 'Rede Rodoviária Clandestina', 'População',
                       'Renda domiciliar per capita', 'Taxa de desemprego', 'Homicídios',
                       'PIB', 'Exportação', 'Analfabetos']]
    y_data = table_csv['Área Devastada']

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