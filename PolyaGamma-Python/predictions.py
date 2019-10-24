
# coding: utf-8
"""
Predictions
"""


"""
Import Libraries
"""
# Libraries
import numpy as np



#FUNCTION: return the logistic regression for multiple betas
#PARAMETERS: X and beta_design are design matrices
def logisticRegression(X, beta_design):
    #check if X and beta_design are matrices
    if (not (np.isarray(X)|np.isarray(X))) | (not (np.isarray(beta_design)|np.isarray(beta_design))):
        #(!(is.matrix(X)|is.vector(X))) | (!(is.matrix(beta_design)|is.vector(beta_design))):
        raise ValueError("Parameters in logisticRegression are not of the correct type")
    #end if
    #get a matrix of etas
    eta = beta_design @ X.T #beta_design %*% t(X);
    #return logistic regression
    p = 1 / (1+ np.exp(-eta))
    return p
#end logisticRegression



#' Predict the y value for all data points,
#' given  and the sample from posterior of betas
#'
#' @param X A design matrix.
#' @param beta Sample from the posterior distribution of betas. Alternatively, could be a vector with estimated betas.
#'
#' @export
def get_predictions(X, beta):
    samples = logisticRegression(X, beta)
    posterior_mean = np.mean(samples, axis=0) #colMeans(samples)
    ypred = round(posterior_mean)
    return ypred



