import sys
import pandas as pd
import numpy as np 
from sklearn import tree

def get_class_count (d):
    D = {}
    for x in d:
       if x not in D:
          D[x] = 1
       else:
          D[x] = D[x] + 1
    return D 


def get_class_split (x, y):
    x_class = get_class_count (x)
    y_class = get_class_count (y)

    num_x_class = len ([x for x in x_class])
    A = []
    for c in x_class.keys():
       B = []
       for j in range(0, len(y)):
          if x[j] == c:
             B.append(y[j])
       A.append(B)
    return A 


  
def get_entropy (d):
    D = get_class_count (d) 
    total = np.sum ([D[x] for x in D])
    e  = 0.0 
    for x in D:
       p = D[x]/total 
       e = e - p * np.log2(p)
    return e 
 

def get_gini (d):
    D = get_class_count (d) 
    total = np.sum ([D[x] for x in D])
    g  = 0.0
    for x in D:
       p = D[x]/total
       g = g + p * p
    return  g

 
def get_gini_split (x,y):
    A = get_class_split (x, y)
    n = [len(A[i]) for i in range(0, len(A))]

    g_sum = 0 
    for i in range (0, len(A)):
       g = get_gini (A[i])
       g_sum += g * float(n[i])/np.sum(n)

    return g_sum 


def get_entropy_split (x, y):
    A = get_class_split (x, y)
    n = [len(A[i]) for i in range(0, len(A))]
    
    e_root = get_entropy (y)
    print("e_root=",e_root)
    e_sum = 0.0 
    for i in range (0, len(A)):
       e = get_entropy (A[i])
       e_sum += e * float(n[i])/np.sum(n)
    return e_sum 


if __name__ == "__main__":

    df = pd.read_csv(sys.argv[1])

    print(df.shape)
    print(df.columns)
  
    x1 = df['is_phd'].tolist()
    x2 = df['hindu'].tolist()
    x3 = df['been_abroad'].tolist()
    x4 = df['hindi'].tolist()
    x5 = df['north'].tolist()
    y = df['is_right'].tolist()


    ex1  = get_entropy (x1) 
    ex2  = get_entropy (x2) 
    ex3  = get_entropy (x3) 
    ex4  = get_entropy (x4) 
    ex5  = get_entropy (x5) 
    ey  = get_entropy (y) 

    print("ENTROPY:", ex1,ex2,ex3,ex4,ex5,ey)

    g = []
 
    g.append(get_gini_split(x1, y))
    g.append(get_gini_split(x2, y))
    g.append(get_gini_split(x3, y))
    g.append(get_gini_split(x4, y))
    g.append(get_gini_split(x5, y))
    print("g=",g)
    print(np.argmax(g))


    print("entropy:")
    e = []

    e.append(get_entropy_split(x1, y))
    e.append(get_entropy_split(x2, y))
    e.append(get_entropy_split(x3, y))
    e.append(get_entropy_split(x4, y))
    e.append(get_entropy_split(x5, y))    
    print("e=",e)
    print(np.argmin(e))
    e_min = np.min (e)
    print("root entropy:", ey, "min entropy:", e_min, "de=", ey-e_min)
