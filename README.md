
## Repositório que contém as análises das simulações do artigo: Amazon rainforest deforestation influenced by clandestine and regular roadway network. 

### Execução dos Scripts de simulação:
#### 1. Instalação do ambiente virtual (opcional)

1.1 - Baixe e instale o miniconda do [site oficial](https://docs.conda.io/en/latest/miniconda.html).
    
1.2 - Crie um ambiente virtual
```bash
    $ conda create -n myenv
```

1.3 - Ative o ambiente virtual criado
```bash
    $ conda activate myenv
```
        
#### 2. Instalação os requisitos
```bash
$ pip install -r requirements.txt
```

#### 3. Definição dos parâmetros:
É necessários editar o arquivo `parameters.json` e indicar os parâmetros desejados:
- `dataset`: caminho que contém o arquivo com os dados a serem simulados.
- `variables_sensibility`: variáveis escolhidas para o teste de sensibilidade.
- `percent_variance`: lista de valores com os percentuais de variação das variáveis de sensibilidade.
- `years_sensibility`: lista de anos (no formato de texto) analisados.
- `classifiers`: lista de modelos de aprendizado a serem utilizados na simulação. Ex:"RF", "DT" e "KNN".
- `path_results`: caminho que conterá os arquivos de saída da simulação.
- `enable_acc`: indica se será utilizado uma análise acumulativa dos dados de área devastada.

#### 4. Executar experimentos
```bash
    $ python main.py
```

#### 5. É possível também realizar uma Análise de Correlação de Pearson da seguinte forma:
```bash
    $ python corr_analisys.py
```
Note que esse script também utiliza como referência o arquivo `parameters.json` para definição de seus parâmetros de execução.

No entanto, neste caso, apenas as variáveis abaixo serão consideradas para efeitos desse tipo de simulação:
- `dataset`
- `path_results`
- `enable_acc`
    
##### Atenção
Após todos esses passos, todos os resultados gerados estarão disponíveis na pasta `path_results` definida no arquivo `parameters.json`.