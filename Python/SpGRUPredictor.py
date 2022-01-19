"""
Created on Wed Mar 31 02:57:52 2021
SPGRUPREDICTOR GRU based sparse predictor
   Code by Fredy Vides
   For Paper, "Computing Sparse Semilinear Models for Approximately Eventually Periodic Signals"
   by F. Vides
@author: Fredy Vides
"""
def SpGRUPredictor(data,model,h,N):
    from numpy import append,reshape
    Lag = len(h)
    md = data.min()
    Md = abs(data - md).max()
    X = []
    x0 = []
    X = h
    x0 = reshape(h.copy(),(1,1,Lag))
    for j in range(N):
        x = model(x0)
        x0[:1,:,:][0][0][:-1]=x0[:1,:,:][0][0][1:]
        x0[:1,:,:][0][0][-1]=x
        X = append(X,x)
    X = Md*X+md    
    return X