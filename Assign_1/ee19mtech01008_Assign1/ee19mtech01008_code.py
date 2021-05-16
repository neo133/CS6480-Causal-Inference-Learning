import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

def causal_effect(X, y, model=LinearRegression(), treatment=0):
    model.fit(X, y)
    X1 = pd.DataFrame.copy(X)
    X1[X.columns[treatment]] = 1
    X0 = pd.DataFrame.copy(X)
    X0[X.columns[treatment]] = 0
    return (model.predict(X1) - model.predict(X0)).mean()

######## given set of equations ###############
w = np.random.normal(65, 5, 10000)
t = (w/18) + np.random.normal(0, 1, 10000)
######## Binarize t ###############
t[t>3.5].astype(int)
###################################
y = (1.05*t) + (2*w) + np.random.normal(0, 1, 10000)
z = (0.4*t) + (0.3*y) + np.random.normal(0,1, 10000)
######### conversion to panda dataframe ##########
df = pd.DataFrame({'w':w,'t':t,'y':y,'z':z})
#################################################
ate=None
ate3=None
ate2=None
ate1=None
ate3 = causal_effect(df[['t']],df['z'], treatment=0)
print("E[Z|T=t] - E[Z|t=t']: ",ate3)
ate =causal_effect(df[['t']], df['y'], treatment=0)
print("E[Y|T=1] - E[Y|T=0]: ",ate)
ate1 = causal_effect(df[['t', 'w']], df['y'], treatment=0)
print("E[Y|do(T=1)] - E[Y|do(T=0)] with W adjustment: ",ate1)
ate2 = causal_effect(df[['t', 'w', 'z']], df['y'], treatment=0)
print("E[Y|do(T=1)] - E[Y|do(T=0)] with W,Z adjustment: ",ate2)
