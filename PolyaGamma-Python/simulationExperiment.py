
# coding: utf-8

"""
Simulation Experiment
"""


"""
Libraries
"""
# Libraries
import numpy as np
import matplotlib.pyplot as plt
##
from SamplerPG import gibbs_sampler



def logistic(x):
    return 1 / (1 + np.exp(-x))

#' Simulate data from a simple logistic model
#' @export
def generate_from_simple_logistic_model(n=1000):
    x = np.random.normal(0,1, size=n).reshape(-1,1) 
    X = np.concatenate((np.ones(n).reshape(-1,1), x), axis=1) 
    y = np.random.binomial(1, logistic(1 + 1*x))
    return {"X": X, "y": y}


data = generate_from_simple_logistic_model(1000)
obj = gibbs_sampler(data['y'], data['X'], b=np.zeros(2), B=10*np.identity(2), n_iter_total=500)


fig = plt.figure(figsize=(9,5))
plt.plot(obj['beta'][:,0])
plt.plot(obj['beta'][:,1], 'g')

plt.show()


#naive_out = gibbs_sampler(data['y'], data['X'], b=np.repeat(0, 2), B=10*np.identity(data['X'].shape[1]), naive=True,n_iter_total=500)
#
#plt.plot(naive_out['beta'][:,0])
#plt.show()
#plt.plot(naive_out['beta'][:,1],'g')