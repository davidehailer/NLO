import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

def plot_Standort(x_Q, S, u):
    dim, numb_sensors = np.shape(S)
    fig, ax1 = plt.subplots(figsize=(13,13))
    plt.axis('equal')
    plt.axis([-20, 20, -25, 20])
    
    for i in range(0,numb_sensors):
        ax1.add_artist(Circle(S[:,i], u[i], color = 'blue', fill = False))
        plt.plot(S[0,i], S[1,i], 'bo')
        ax1.annotate('s{}'.format(i+1), xy=(S[0,i]-0.4, S[1,i]+0.7), size=15)
        
    plt.plot(x_Q[0],x_Q[1], 'ro', label = 'x_prediction')
    plt.legend(prop={'size': 15})
    plt.title("Lokalisierung einer Signalquelle", size=18)
    plt.show()