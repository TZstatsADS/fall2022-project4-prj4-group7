# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 19:54:21 2022

@author: ssk2258
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize # for loss func minimization
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

#################### A3 ####################

# def logistic_loss(theta, X, y, return_arr=None):

#     tx = np.dot(X,theta).flatten()
#     yz = np.multiply(y,tx)

#     if return_arr == True:
#         out = -(log_logistic(yz))
#     else:
#         out = -np.sum(log_logistic(yz))
#     return out

def loss_function(w,X,y,return_arr = None):
    yz = y*np.dot(X,w)
    out = np.empty_like(yz)
    ind = yz >0.0
    out[ind] = -np.log(1.0+np.exp(-yz[ind]))
    out[~ind] = yz[~ind] - np.log(1.0+np.exp(yz[~ind]))
    if return_arr == True:
        return -out
    else:
        return -np.sum(out)

# def log_logistic(x):

# 	out = np.empty_like(x) # same dimensions and data types

# 	i = x>0
# 	out[i] = -np.log(1.0 + np.exp(-x[i]))
# 	out[~i] = x[~i] - np.log(1.0 + np.exp(x[~i]))
# 	return out

# def accuracy(w, x, y):
#     shape = x.shape[1]
#     pred = np.dot(x, w.reshape(shape,1))
#     pred_prob = 1/(1+ np.exp(-pred))
    
#     pred_prob[pred_prob>=0.5] = 1
#     pred_prob[pred_prob<0.5] = -1
    
#     matches = np.where(pred_prob == y)
    
#     return (matches[0].shape[0]/pred_prob.shape[0]), pred_prob

# def get_calibration(y_pred, y_true,x_sensitive):
#     y_pred = y_pred.astype('float64')
#     y_true = y_true.astype('float64')
#     x_sensitive = x_sensitive.astype('float64')
#     idx = (x_sensitive == 0)
#     p1 = np.mean(y_pred[idx]==y_true[idx])
#     p0 = np.mean(y_pred[~idx]==y_true[~idx])
#     out = p1-p0
#     return out

# def p_rule(x_sensitive, y_pred):
    
#     not_protected = np.where(x_sensitive != 0)[0]
#     protected = np.where(x_sensitive == 0)[0] 
    
#     protected_preds = np.where(y_pred[protected] == 0)
#     nonpro_preds = np.where(y_pred[not_protected] == 0)
#     protected_perc = (protected_preds[0].shape[0]/protected.shape[0]) 
#     nonpro_perc = (nonpro_preds[0].shape[0]/not_protected.shape[0])
    
#     perc_ratio = protected_perc/nonpro_perc
    
#     return perc_ratio, protected_perc, nonpro_perc

def calibration(y_pred,y_actual,s):
    cal_0 = 0
    cal_1 = 0
    count_0 = 0
    count_1 = 0
    for i in range(len(y_pred)):
        if s[i] == 0:
            count_0 += 1
            if list(y_actual)[i] == y_pred[i]:
                cal_0 += 1
        else:
            count_1 += 1
            if list(y_actual)[i] == y_pred[i]:
                cal_1 += 1

    cal_0 /= count_0
    cal_1 /= count_1

    return cal_0 - cal_1

def model_gamma(x, y, s,theta_star, gamma=0.1):

    unconstrained_loss_arr = loss_function(theta_star, x, y, return_arr=True)

    def constraint_gamma_all(theta, x, y,  initial_loss_arr):
        
        new_loss = loss_function(theta, x, y)
        old_loss = sum(initial_loss_arr)
        return ((1.0 + gamma) * old_loss) - new_loss

    constraints = [{'type': 'ineq', 'fun': constraint_gamma_all, 'args':(x,y,unconstrained_loss_arr)}]

    #Covariance function for Optmization
    def opt_function(w, x, x_sensitive):
        protected_cov = (x_sensitive - np.mean(x_sensitive)) * np.dot(w, x.T)
        return float(abs(sum(protected_cov))) / float(x.shape[0])

    theta = minimize(fun = opt_function,
    x0 = theta_star,
    args = (x, s),
    method = 'SLSQP',
    options = {"maxiter":100000},
    constraints = constraints
    )

    return theta.x

def predict(w,x):
    z = np.dot(w,x.T)
    y = 1/(1+np.exp(-z))
    y = (y>=0.5)
    y = y.astype('float64')
    return y

def model_fg(x, y, s, theta_star, gamma=0.1):

    constraints = []
    predicted_labels = predict(theta_star,x)
    unconstrained_loss_arr = loss_function(theta_star, x, y, return_arr=True)

    def constraint_protected_people(w,x,y):
        #for fine-gamma, constraint to prevent non-protected user be classify as negative
        return(np.dot(w,x.T))
    
    def constraint_unprotected_people(w,old_loss,x,y):
        new_loss = loss_function(w,np.array([x]),np.array(y))
        return((1.0+gamma)*old_loss)-new_loss

    for i in range(0, len(predicted_labels)):
        if predicted_labels[i] == 0 and s[i] == 0:
            c = {'type': 'ineq', 'fun': constraint_protected_people, 'args':(x[i], y[i])}
            constraints.append(c)
        else:
            c = {'type': 'ineq', 'fun': constraint_unprotected_people, 'args':(unconstrained_loss_arr[i], x[i], y[i])}
            constraints.append(c)
    
    def opt_function(w, x, x_sensitive):
      protected_cov = (x_sensitive - np.mean(x_sensitive)) * np.dot(w, x.T)
      return float(abs(sum(protected_cov))) / float(x.shape[0])

    theta = minimize(fun = opt_function,
        x0 = theta_star,
        args = (x, s),
        method = 'SLSQP',
        options = {"maxiter":100000},
        constraints = constraints
        )

    return theta.x


#################### A6 ####################

def partition(X,e):
    e_list = list(e.keys())
    X_i = []
    for i in range(len(e_list)):
        X[e_list[i]] = e[e_list[i]]
        X_i.append(X)
    return X_i

def ceildiv(a, b):
    return -(a // -b)

def delta(item, y_pred, y_prob, G_i = 12):
    
    item_copy = item.copy()
    item_copy['label'] = y_pred
    af = item_copy[item_copy.race=='African-American']
    ca = item_copy[item_copy.race =='Caucasian']
    rate_af = round(af[af['label']==1].shape[0]/af.shape[0],2)
    rate_ca = round(ca[ca['label']==1].shape[0]/ca.shape[0],2)
    p_star = (rate_af+rate_ca)/2
    threshold = np.abs(rate_af-p_star)
    
    test_df = pd.DataFrame(y_prob,columns=['No','Yes'])
    test_df.set_index(item.index,inplace=True)
    test_df['race']=item['race']
    test_df['label'] = y_pred
    temp = test_df[(np.abs(test_df.Yes - test_df.No)<=G_i*threshold)]
#     print(temp.shape[0],threshold)
    for i in range(temp.shape[0]):
        if temp.race.iloc[i] == 'African-American' and temp.Yes.iloc[i] > temp.No.iloc[i]:
            temp.label.iloc[i] = 0
        elif temp.race.iloc[i] == 'Caucasian' and temp.Yes.iloc[i] < temp.No.iloc[i]:
            temp.label.iloc[i] = 1
    
    item_copy['label'].loc[temp.index] = temp.label
    
    return item_copy

def delta2(item, y_pred, y_prob, G_i = 6):
    
    item_copy = item.copy()
    item_copy['label'] = y_pred
    af = item_copy[item_copy.race=='African-American']
    ca = item_copy[item_copy.race =='Caucasian']
    rate_af = round(af[af['label']==1].shape[0]/af.shape[0],2)
    rate_ca = round(ca[ca['label']==1].shape[0]/ca.shape[0],2)
    p_star = (rate_af+rate_ca)/2
    threshold = np.abs(rate_af-p_star)
    
    test_df = pd.DataFrame(y_prob,columns=['No','Yes'])
    test_df.set_index(item.index,inplace=True)
    test_df['race']=item['race']
    test_df['label'] = y_pred
    temp = test_df[(np.abs(test_df.Yes - test_df.No)<=G_i*threshold)]
#     print(temp.shape[0],threshold)
    temp_dic = temp[['race','label']].value_counts().to_dict()
    
    aa_0,aa_1,ca_0,ca_1 = 0,0,0,0
    for key, value in zip(temp_dic.keys(),temp_dic.values()):
        if key == ('African-American', 1):
            aa_1 = ceildiv(value,2)
        if key == ('African-American', 0):
            aa_0 = ceildiv(value,2)       
        if key == ('Caucasian', 1):
            ca_1 = ceildiv(value,2)        
        if key == ('Caucasian', 0):
            ca_0 = ceildiv(value,2)

    if aa_1>aa_0:
        aa_replace = aa_0
    else:
        aa_replace = aa_1

    if ca_1>ca_0:
        ca_replace = ca_0
    else:
        ca_replace = ca_1

    count_aa = 0
    count_ca = 0

    while(count_aa<aa_replace):
        for idx in list(temp.index):
            if temp.race.loc[idx] == 'African-American' and temp.label.loc[idx] ==1:
                temp.label.loc[idx] = 0
                count_aa +=1

    while(count_ca<ca_replace):
        for idx in list(temp.index):
            if temp.race.loc[idx] == 'Caucasian' and temp.label.loc[idx] ==0:
                temp.label.loc[idx] = 1
                count_ca +=1
    
    item_copy['label'].loc[temp.index] = temp.label
    
    return item_copy

def local_massaging(X,s,e,y):
    X['race'] = s
    X_i_list =partition(X,e)
    df_list = []
    pd_list = []
    for item in X_i_list:
        enc = OneHotEncoder(handle_unknown='ignore')
        X_new = enc.fit(item)
        X_new = enc.transform(item)
        X_train, X_test, y_train, y_test = train_test_split(X_new,y,test_size=.3,random_state=5)
        forest = RandomForestClassifier(random_state=44).fit(X_train, y_train) 
        y_pred = forest.predict(X_new)
        y_prob = forest.predict_proba(X_new)
        temp_df = delta(item,y_pred,y_prob)
        pd_list.append(temp_df)
        df_list.append(temp_df[['label']])
    
    # Here we take vote 
    
    result = pd.concat(df_list,axis = 1)
    result.loc[result.sum(axis=1)<=2,'new_label'] = 0
    result.loc[result.sum(axis=1)>=3,'new_label'] = 1
    X['new_label'] = result.new_label
    return X

def local_preferential_sampling(X,s,e,y):
    X['race'] = s
    X_i_list =partition(X,e)
    df_list = []
    pd_list = []
    for item in X_i_list:
        enc = OneHotEncoder(handle_unknown='ignore')
        X_new = enc.fit(item)
        X_new = enc.transform(item)
        X_train, X_test, y_train, y_test = train_test_split(X_new,y,test_size=.3,random_state=42)
        forest = RandomForestClassifier(random_state = 88).fit(X_train, y_train) 
        y_pred = forest.predict(X_new)
        y_prob = forest.predict_proba(X_new)
        temp_df = delta2(item,y_pred,y_prob,6)
        pd_list.append(temp_df)
        df_list.append(temp_df[['label']])
    
    # Here we take vote 
    
    result = pd.concat(df_list,axis = 1)
    result.loc[result.sum(axis=1)<=2,'new_label'] = 0
    result.loc[result.sum(axis=1)>=3,'new_label'] = 1
    X['new_label'] = result.new_label
    
    return X






