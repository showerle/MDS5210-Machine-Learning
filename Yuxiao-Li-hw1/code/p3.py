# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 16:58:25 2021

@author: zhezi

Modified on Thu March 17 17:53:28 2022

@author: qijun
"""


###################   MDS5210 Machine Learning HW # 1   ########################

###################  Question 4 (d) ########################
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def huber_grad(z, mu):
    #evaluate the 1st order deriative of huber

    #grad = sign(z) * (abs(z)>=mu) + (z/mu) * (abs(z)<mu)
    
    grad = np.sign(z) * (abs(z)>=mu) + (z/mu) * (abs(z)<mu)
    
    return grad

def GD_Huber(X, y, theta_0, mu, alpha, maxIter, theta_star):
    #initialization the algorithm
    theta = theta_0
    #main loop
    Error_huber = []
    for i in range(1,maxIter):
    #calculate error
        Error_huber.append(np.linalg.norm(theta-theta_star, 2)) 
        
    
    #calculate gradient
        residual = X.dot(theta) - y
        grad = X.T.dot(huber_grad(residual,mu)) 
    
    #gradient descent update
        theta = theta - alpha * grad
        
    theta_huber = theta
    
    return Error_huber, theta_huber

#######   generate date   ########
n = 1000  # number of samples
d = 50   #feature dimension
ratio = 0.3  #Outliers ratio 
sigma = 0.1  # variance of Gaussian noise

X = np.load('X.npy')
y = np.load('y.npy')
theta_star = np.load('theta_star.npy')

# approxing L_1 robust regreesion approximated by Huber norm

# set parameters
mu = 1e-5  #the smoothing parameter for huber approximation
alpha = 0.001  #stepsize for runing gradient descent
maxIter = 1000  #iteration number
theta_0 = np.random.randn(d,1)  #random initialization for gradient descent

#  =======run gradient descent on the Huber approximation=======
Error_huber_result, theta_huber_result = GD_Huber(X, y, theta_0, mu, alpha, maxIter, theta_star) 


###### LS estimator ########

A = np.linalg.inv(X.T.dot(X))
theta_hat = (A.dot(X.T)).dot(y)#closed form solution, have a look at our slides

Error_LS = np.linalg.norm(theta_hat - theta_star, 2)
print('Estimator approximated by LS:',Error_LS)

#######   plot the figure   #########
plt.figure(figsize=(10,5))
plt.yscale('log',base=2) 
plt.plot(Error_huber_result, 'r-')
plt.title(r'$\ell_1$ estimator approximated by Huber')
plt.ylabel(r'$\theta$')               # set the label for the y axis
plt.xlabel('Iteration')              # set the label for the x axis
plt.show()