import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

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
 

def main():
    var = 'set'
    df = pd.read_csv('dados/variaveis_' + var + '_2020_pa.csv')
    df = df.drop(columns=['ano'])
    ACC = True
    path_out = 'resultados_' + var
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    cols = []
    for col in  columns_format.values():
        cols.append(col)

    df.columns = cols

    if ACC:
        x_0 = 267393.0285 - df['DA'][:-1].sum()
        val_acc = []
        for i, val in enumerate(df['DA']):
            val_acc.append(sum(df['DA'][:i+1]) + x_0)
        
        df['DA'] = val_acc

    corr = df.corr()
    
    f, ax = plt.subplots(figsize=(9, 6))
    
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask, 1)] = True
    sns.heatmap(corr, mask=mask, cmap='RdBu', annot=True, linewidths=.5, ax=ax)
    plt.savefig(path_out + '/' + 'correl_' + var + '.png')  
    plt.show()


if __name__ == '__main__':

    main()
