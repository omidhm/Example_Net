
def t(Kfast, Avg):
    import numpy as np
    T = np.random.uniform(0,1,Avg)
    Time_Index  = sorted(range(len(T)), key = lambda kk: T[kk])
    Temp = Time_Index[0:Kfast]
    Out = T[Temp[Kfast-1]] 
    return Out, Temp