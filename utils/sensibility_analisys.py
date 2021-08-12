import pandas as pd
from utils import *


def sensibility_analysis(x, y, i, p, model, variable, df_sens):
    '''
        Computes the sensiblity of the modelo by varianting one just variable.

        Parameters
        ----------
            x: (np.narray) --> 1xN
                Sample of the dataset
            
            y: (np.ndarray) --> 1x1
                Desired value for the sample x.
            
            i: (int)
                Index of the analized variable.
            
            p: (float)
                Percent of the variation on the analized sample.
            
            model: (RandomForestRegressor)
                Object of the class RandomforestRegressor previously trained.
            
            variable: (string)
                Representation of the analized variable.
            
            df_sens: (pd.DataFrame)
                Dataframe object that storage the results.
        Return
        -------
            df_sens: (pd.DataFrame)
                Dataframe object that storage the results updated.
    '''

    x_down = x.copy()
    x_up = x.copy()
    
    x_up[i]   *= (1 + p) # increase +p%
    x_down[i] *= (1 - p) # decrease -p%

    y_up   = model.predict([x_up])
    y_pred = model.predict([x])
    y_down = model.predict([x_down])
    
    var_x  = abs(x[i] - x_up[i]) # or abs(x[i] - x_down[i])

    var_y_up   = abs(y - y_up)[0]
    var_y_down = abs(y - y_down)[0]

    efect_y_up   = var_y_up / var_x
    efect_y_down = var_y_down / var_x

    df_sens = df_sens.append({variable: x_down[i], 
                            'DA PREDICT':  y_down[0], 
                            variable +' VAR': var_x, 
                            'DA VAR': var_y_down, 
                            'EFFECT': efect_y_down}, 
                            ignore_index=True)
    
    df_sens = df_sens.append({variable: x[i],
                            'DA TRUE': y,
                            'DA PREDICT': y_pred[0],
                            variable +' VAR': None,
                            'DA VAR': None,
                            'EFFECT': None},
                            ignore_index=True)
    
    df_sens = df_sens.append({variable: x_up[i],   
                            'DA PREDICT':  y_up[0], 
                            variable +' VAR': var_x, 
                            'DA VAR': var_y_up, 
                            'EFFECT': efect_y_up},
                            ignore_index=True)

    return df_sens

def main():
    pass


if __name__ == '__main__':

    main()
