# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 19:54:21 2022

@author: ssk2258
"""

import numpy as np
from scipy.optimize import minimize # for loss func minimization
from copy import deepcopy

def logistic_loss(theta, X, y, return_arr=None):

    tx = np.dot(X,theta).flatten()
    yz = np.multiply(y,tx)

    if return_arr == True:
        out = -(log_logistic(yz))
    else:
        out = -np.sum(log_logistic(yz))
    return out

def log_logistic(x):

	out = np.empty_like(x) # same dimensions and data types

	i = x>0
	out[i] = -np.log(1.0 + np.exp(-x[i]))
	out[~i] = x[~i] - np.log(1.0 + np.exp(x[~i]))
	return out

def calibration(features,y_pred,y_actual):
    cal_0 = 0
    cal_1 = 0
    count_0 = 0
    count_1 = 0
    for i in range(len(features)):
        if list(features["race"])[i] == 0:
            count_0 += 1
            if list(y_actual)[i] == y_pred[i]:
                cal_0 += 1
        else:
            count_1 += 1
            if list(y_actual)[i] == y_pred[i]:
                cal_1 += 1

    cal_0 /= count_0
    cal_1 /= count_1

    return cal_1 - cal_0

def model_gamma(x, y, s, y_pred):

    # train on just the loss function
    theta = minimize(fun = logistic_loss,
                 x0 = np.random.rand(x.shape[1],),
                 args = (x, y),
                 method = 'SLSQP',
                 options = {"maxiter":100000},
                 constraints = []
                 )
    theta_star = deepcopy(theta.x)
    
    gamma = 0.1

    def constraint_gamma_all(theta, x, y,  initial_loss_arr):
        
        new_loss = logistic_loss(theta, x, y)
        old_loss = sum(initial_loss_arr)
        return ((1.0 + gamma) * old_loss) - new_loss

    unconstrained_loss_arr = logistic_loss(theta.x, x, y, return_arr=True)

    constraints = [({'type': 'ineq', 'fun': constraint_gamma_all, 'args':(x,y,unconstrained_loss_arr)})]

    def cross_cov_abs_optm_func(weight_vec, x_in, x_control_in_arr):
        cross_cov = (x_control_in_arr - np.mean(x_control_in_arr)) * np.dot(weight_vec, x_in.T)
        return float(abs(sum(cross_cov))) / float(x_in.shape[0])


    theta = minimize(fun = cross_cov_abs_optm_func,
        x0 = theta_star,
        args = (x, s),
        method = 'SLSQP',
        options = {"maxiter":100000},
        constraints = constraints
        )

    try:
        assert(theta.success == True)
    except:
        print("Optimization problem did not converge.. Check the solution returned by the optimizer.")
        print("Returned solution is:")
        print(theta)

    return theta.x

def model_fg(x, y, s, y_pred):
    
    # train on just the loss function
    theta = minimize(fun = logistic_loss,
                 x0 = np.random.rand(x.shape[1],),
                 args = (x, y),
                 method = 'SLSQP',
                 options = {"maxiter":100000},
                 constraints = []
                 )
    theta_star= deepcopy(theta.x)

    gamma = 0.1

    def constraint_protected_people(theta,x,y): # dont confuse the protected here with the sensitive feature protected/non-protected values -- protected here means that these points should not be misclassified to negative class
        return np.dot(theta, x.T) # if this is positive, the constraint is satisfied

    def constraint_unprotected_people(theta,ind,old_loss,x,y):

        new_loss = logistic_loss(theta, np.array([x]), np.array(y))
        return ((1.0 + gamma) * old_loss) - new_loss

    constraints = []
    predicted_labels = y_pred
    unconstrained_loss_arr = logistic_loss(theta.x, x, y, return_arr=True)
    
    for i in range(0, len(predicted_labels)):
        if predicted_labels[i] == 1.0 and s[i] == 1.0: # for now we are assuming just one sensitive attr for reverse constraint, later, extend the code to take into account multiple sensitive attrs
            c = ({'type': 'ineq', 'fun': constraint_protected_people, 'args':(x[i], y[i])}) # this constraint makes sure that these people stay in the positive class even in the modified classifier             
            constraints.append(c)
        else:
            c = ({'type': 'ineq', 'fun': constraint_unprotected_people, 'args':(i, unconstrained_loss_arr[i], x[i], y[i])})                
            constraints.append(c)

    def cross_cov_abs_optm_func(weight_vec, x_in, x_control_in_arr):
        cross_cov = (x_control_in_arr - np.mean(x_control_in_arr)) * np.dot(weight_vec, x_in.T)
        return float(abs(sum(cross_cov))) / float(x_in.shape[0])

    theta = minimize(fun = cross_cov_abs_optm_func,
        x0 = theta_star,
        args = (x, s),
        method = 'SLSQP',
        options = {"maxiter":100000},
        constraints = constraints
        )

    try:
        assert(theta.success == True)
    except:
        print("Optimization problem did not converge.. Check the solution returned by the optimizer.")
        print("Returned solution is:")
        print(theta)

    return theta.x