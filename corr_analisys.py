import matplotlib.pyplot as plt
import pandas as pd

columns_format = {
    'rede_rodoviaria': ['Rede Roviária', 'RR'],
    'rede_rodoviaria_clandestina': ['Rede Rodoviária Clandestina', 'RDC'],
    'area_devastada': ['Área Devastada', 'AD'],
    'população': ['População', 'POP'],
    'renda_domiciliar_per_capita': ['Renda Domiciliar Per Capita', 'RDPC'],
    'taxa_desemprego': ['Taxa de Desemprego', 'TD'],
    'homicidios': ['Homicídios', 'HOM'],
    'PIB': ['PIB', 'PIB'],
    'exportacao': ['Exportação', 'EXP'],
    'analfabetos': ['Analfabetos', 'ANA']
}


def main():
    
    df = pd.read_csv('variaveis_jan_2020_pa.csv')
    df = df.drop(columns=['ano'])
    print(df.columns)
    print(columns_format.keys())

    quit()

    cols = []    
    for _, col in  columns_format.values():
        cols.append(col)

    df.columns = cols
    corr = df.corr()
    
    
    '''
    plt.matshow(corr, cmap='Spectral', interpolation='none')  
    plt.colorbar()
    plt.xticks(range(len(corr)), cols)
    plt.yticks(range(len(corr)), cols)
    plt.show()
    '''
    import seaborn as sns
    import numpy as np

    f, ax = plt.subplots(figsize=(9, 6))
    
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask, 1)] = True
    sns.heatmap(corr, mask=mask, cmap='RdBu', annot=True, linewidths=.5, ax=ax)
    plt.savefig("correl.png")  
    plt.show()

if __name__ == '__main__':

    main()
