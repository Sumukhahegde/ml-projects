import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import numpy as np


line_width=3

font = {'family' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = line_width


def plot_fit(x, y, y1, label_data, label_model):
 
   fig = plt.figure(figsize=(12,9))
   ax = fig.add_subplot(111)

   ax.set_xlabel('x')
   ax.set_ylabel('y')

   ax.plot(x, y,'o',label=label_data,linewidth=line_width)
   ax.plot(x, y1,label=label_model,linewidth=line_width)

   ax.legend()
   plt.title("Model fitting")
   plt.savefig("model_fit.png")


def plot_params (df):

   fig = plt.figure(figsize=(12,12))

   ax = fig.add_subplot(211)
   bx = fig.add_subplot(212)

   ax.set_xlabel('time')
   ax.set_ylabel('w')

   bx.set_xlabel('time')
   bx.set_ylabel('b')

   ax.plot(df['time'],df['w'],linewidth=line_width) 
   ax.plot(df['time'],df['w'],'.')

   bx.plot(df['time'],df['b'],linewidth=line_width) 
   bx.plot(df['time'],df['b'],'.')

   plt.suptitle("Parameters")
   plt.savefig("params.png")


def plot_perfmn (df):

   fig = plt.figure(figsize=(12,12))

   ax = fig.add_subplot(211)
   bx = fig.add_subplot(212)

   ax.set_xlabel('time')
   ax.set_ylabel('Loss')

   bx.set_xlabel('time')
   bx.set_ylabel('R2')

   bx.axhline(y=0,c='k',ls='--')
   ax.set_yscale('log')
 
   ax.plot(df['time'],df['loss'],linewidth=line_width)
   bx.plot(df['time'],df['R2'],linewidth=line_width)
   plt.suptitle("Performance metrices")

   plt.savefig("perf.png")


def surf_plot (x, y,w_min,w_max,b_min,b_max):

   w = np.linspace(w_min,w_max,n)
   b = np.linspace(b_min,b_max,n)

   result = loss(w[:,None], b[None,:], x, y)




def line (w, b, x):
  return w * x + b


