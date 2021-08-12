import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

# english version
columns_format = {
    'rede_rodoviaria': 'RN',
    'rede_rodoviaria_clandestina': 'CRN',
    'area_devastada': 'DA',
    'população': 'POP',
    'renda_domiciliar_per_capita': 'HIPC',
    'taxa_desemprego': 'UR',
    'homicidios': 'HOM',
    'PIB': 'GDP',
    'exportacao': 'EXP',
    'analfabetos': 'ILL'
}

# portguese version
columns_format = {
    'rede_rodoviaria': 'RR',
    'rede_rodoviaria_clandestina': 'RRC',
    'area_devastada': 'AD',
    'população': 'POP',
    'renda_domiciliar_per_capita': 'RDPC',
    'taxa_desemprego': 'TD',
    'homicidios': 'HOM',
    'PIB': 'PIB',
    'exportacao': 'EXP',
    'analfabetos': 'ANA'
}


def main():
    var = 'jan_novo' #jan=2018, set=2019 #jan_novo #jan_novo_ii
    df = pd.read_csv('dados/variaveis_jan_novo_2020_pa.csv')
    #df = pd.read_csv('variaveis_julho_2021_pa.csv')

    df = df.drop(columns=['ano'])

    if var == 'julho':
        ACC=False
    else:
        ACC=True

    path_out = 'resultados_' + var
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    cols = []
    for col in  columns_format.values():
        cols.append(col)

    df.columns = cols

    if ACC:
        if var == 'jan' or var == 'jan_novo' or var=='jan_novo_ii':
            x_0 = 267393.0285 - df['AD'].sum()

        elif var == 'set':
            x_0 = 267393.0285 - df['AD'][:-1].sum() # pegando valores ate 2018
            
        val_acc = []
        for i, val in enumerate(df['AD']):
            val_acc.append(sum(df['AD'][:i+1]) + x_0)
        
        df['AD'] = val_acc

    corr = df.corr()
    
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(corr, cmap='RdBu', annot=True, linewidths=.5, ax=ax)
    plt.savefig(path_out + '/' + 'correl_' + var + '.png')  
    plt.show()


if __name__ == '__main__':

    main()
