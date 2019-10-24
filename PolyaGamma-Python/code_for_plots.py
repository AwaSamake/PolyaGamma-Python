
# coding: utf-8

"""
Plots 
"""


"""
Import Libraries
"""
# Libraries
import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt
##
from SamplerPG import *


# 
# Prior, likelihood and posterior plot

if(FALSE){
  library(mvtnorm)
  library(ggplot2)
  library(gridExtra)

  n = 100
  x1 = rnorm(n)
  x2 = rnorm(n)
  psi = 1*x1 + 1*x2
  y = rbinom(n, 1, 1 / (1 + exp(-psi)))
  plot(x1, x2, col=y+1, pch=16)


  Xnorm1 = rmvnorm(20000, mean=c(0, 0), sigma = 0.2*diag(2))
  dfnorm1 = data.frame(Xnorm1)
  Xnorm2 = rmvnorm(20000, mean=c(0, 0), sigma = 0.03*diag(2))
  dfnorm2 = data.frame(Xnorm2)

  X = cbind(x1, x2)
  obj = gibbs_sampler(y, X, lambda=0, n_iter_total=1050, burn_in=50)
  df1 = data.frame(obj$beta)
  obj = gibbs_sampler(y, X, lambda=1, n_iter_total=1050, burn_in=50)
  df2 = data.frame(obj$beta)
  obj = gibbs_sampler(y, X, lambda=10, n_iter_total=1050, burn_in=50)
  df3 = data.frame(obj$beta)


  xlim = c(-1, 2)
  ylim = c(-1, 2)

  p1 = ggplot() +
    stat_density2d(aes(X1, X2), dfnorm1, col="black") +
    stat_density2d(aes(X1, X2), df1) +
    stat_density2d(aes(X1, X2), df2, col="red") +
    theme_classic() +
    geom_vline(xintercept=0) + geom_hline(yintercept=0) +
    coord_cartesian(xlim=xlim, ylim=ylim) +
    ggtitle("(a)") + xlab(expression(beta[1])) + ylab(expression(beta[2]))

  p2 = ggplot() +
    stat_density2d(aes(X1, X2), dfnorm2, col="black") +
    stat_density2d(aes(X1, X2), df1) +
    stat_density2d(aes(X1, X2), df3, col="red") +
    theme_classic() +
    geom_vline(xintercept=0) + geom_hline(yintercept=0) +
    coord_cartesian(xlim=xlim, ylim=ylim) +
    ggtitle("(b)") + xlab(expression(beta[1])) + ylab(expression(beta[2]))

  grid.arrange(p1, p2, ncol=2)
}

# Plot with boxplots comparing our implementation and the BayesLogit package,
# together with MSE

if(FALSE){
  data = generate_from_simple_logistic_model(n=2000)

  library(BayesLogit)
  library(ggplot2)
  library(reshape2)
  library(dplyr)
  library(gridExtra)

  simulation_experiment = function(n_values, n_iter = 1000, lambda=100){
    m = length(n_values)
    mse = function(x, y)(mean((x - y)**2))

    chains1 = matrix(NA, n_iter, m)
    chains2 = matrix(NA, n_iter, m)

    set.seed(1)
    for(i in 1:m){
      ind = sample(1:length(data$y), n_values[i])
      obj_bayeslogit = logit(data$y[ind], data$X[ind, ], P0 = lambda*diag(2), samp=n_iter, burn=100)
      obj = gibbs_sampler(data$y[ind], data$X[ind, ], lambda=lambda, n_iter_total=n_iter+100, burn_in=100)
      chain1 = obj$beta[,2]
      chain2 = obj_bayeslogit$beta[,2]


      chains1[, i] = chain1
      chains2[, i] = chain2
    }
    colnames(chains1) = n_values
    colnames(chains2) = n_values
    chains1 = as.data.frame(chains1) %>%
      mutate(type="our") %>%
      melt()
    chains2 = as.data.frame(chains2) %>%
      mutate(type="BayesLogit") %>%
      melt()
    df = rbind(chains1, chains2)

    return(df)
  }

  n_values = c(100, 250, 500, 1000, 1500, 2000)
  df1 = simulation_experiment(n_values, n_iter = 1000, lambda = 0.01)
  df2 = simulation_experiment(n_values, n_iter = 1000, lambda = 10)

  dfmse1 = df1 %>%
    group_by(type, variable) %>%
    summarise(mse = mse(value, 1))

  dfmse2 = df2 %>%
    group_by(type, variable) %>%
    summarise(mse = mse(value, 1))

  p1 = ggplot(df1) +
    geom_boxplot(aes(variable, value, fill=type)) +
    geom_hline(yintercept=1, linetype="dashed")+
    theme_classic() +
    theme(legend.position="none") +
    scale_fill_brewer(palette="Set1") +
    xlab("sample size") + ylab(expression(beta[1]))

  p2 = ggplot(df2) +
    geom_boxplot(aes(variable, value, fill=type)) +
    geom_hline(yintercept=1, linetype="dashed")+
    theme_classic() +
    theme(legend.position="none") +
    scale_fill_brewer(palette="Set1") +
    xlab("sample size") + ylab(expression(beta[1]))


  p3 = ggplot(dfmse1) +
    geom_line(aes(variable, mse, col=type, group=type)) +
    theme_classic() + ylab("MSE") +
    theme(legend.position="none") +
    scale_color_brewer(palette="Set1") +
    xlab("sample size")

  p4 = ggplot(dfmse2) +
    geom_line(aes(variable, mse, col=type, group=type)) +
    theme_classic() + ylab("MSE") +
    theme(legend.position="none") +
    scale_color_brewer(palette="Set1") +
    xlab("sample size")

  grid.arrange(p1, p2, p3, p4, layout_matrix = cbind(c(1, 1, 3), c(2, 2, 4)))

}



