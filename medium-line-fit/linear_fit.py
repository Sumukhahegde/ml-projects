import sys
import numpy as np
import pandas as pd
import argparse 
from pltutils import plot_fit, plot_params, plot_perfmn

np.random.seed(seed=202)

def line (w, b, x):
  return w * x + b  


def get_data ():

   n = 100 # number of data points 
   weight = 1.81
   bias = 34.0
   
   # for noise 
   mu = 0.0
   sigma = 2.0
 
   x = np.linspace (0, 10, n)
   y = line (weight, bias, x) + np.random.normal(mu,sigma,n)

   return weight, bias, mu, sigma, x, y 


def R2 (w, b, x, y):
   ybar = np.mean (y) 

   y1 = line (w, b, x)

   S1 = np.sum ( (y1-ybar)**2.0)
   S2 = np.sum ( (y-ybar)**2.0)

   return 1 - S1/S2 

  
if __name__ == "__main__":

   parser = argparse.ArgumentParser()
   parser.add_argument('-n','--niter',type=int,default=1000,\
      help='Number of iterations')
   parser.add_argument('-a','--alpha',type=float,default=0.01,\
      help='Number of Learning rate')

   args = parser.parse_args()

   W, B, mu, sigma, x, y = get_data ()
   n = y.shape[0]


   # starting guess  
   w, b = 1.0, 60.0 

   # we will write all this in a csv file 
   df = pd.DataFrame(columns=['time','loss','R2','w','w_grad','b','b_grad'])

   for i in range (0, args.niter):
      # predict 
      y1 = line (w, b,  x)
      # get the error 
      e = y - y1 
 
      # solve for gradient 
      S,S2,SX = 0.0, 0.0,0.0
      for j in range (0, n):
        S += e[j] 
        S2 += e[j] * e[j]    
        SX += e[j] * x[j]  

      grad_w = -2 * SX /n 
      grad_b = -2 * S/ n

      #update the weight and bias 

      w  = w - args.alpha * grad_w 
      b  = b - args.alpha * grad_b 

      # get r2
      r2 =  R2 (w, b, x, y)

      # write output 
      output = [i, S2,r2,w, grad_w, b, grad_b]
      print(output)
      df.loc[i] = output 

   print("w=",w,"b=",b,"loss=",S2)
 
   df.to_csv("output.csv")


   y1 = line (w, b, x)
   label_data = 'Data [w='+str(W)+", b="+str(B) + ", $\mu$="+str(mu)+", $\sigma$="+str(sigma)+"]"
   label_model = 'Model [w='+str("%.2f" %w)+", b="+str("%.2f" %b)+"]"
   plot_fit(x, y,y1, label_data,label_model)
   
   plot_params (df)
   plot_perfmn (df)
