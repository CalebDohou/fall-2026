#---------------------------------------------------
# ECON 6343: Econometrics III — Problem Set 7 (GMM and SMM)
# Caleb Dohou
#
# Script file: load packages and the source file, then run everything.
# The writeup (answers to all questions) is in the comments at the bottom.
#---------------------------------------------------
using Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM

cd(@__DIR__)

Random.seed!(1234)

# Read in the functions
include("PS7_Dohou_source.jl")

# Run the main function (Question 6: everything is wrapped in main())
results = main();

#---------------------------------------------------
# Question 1: OLS via GMM
#---------------------------------------------------
# Moment function: g = y - Xβ, and the objective is g'Wg with W = I, so g'g.
# GMM estimates: [1.8691, -0.0049, 0.1264, 0.4237]
# OLS estimates: [1.8691, -0.0049, 0.1264, 0.4237]
# They are the same (difference is 3.8e-10), so the GMM code is correct.

#---------------------------------------------------
# Question 2: Multinomial logit by MLE and GMM
#---------------------------------------------------
# (a) and (b): the GMM estimates with MLE starting values are close to the
# MLE but not exactly the same, because GMM with W = I is a different
# estimator. GMM objective = 1699.8.
#
# (c): with random starting values we get very different estimates
# (for example 3.96 for age) and the objective is 3840, much bigger than
# 1699.8. Actually 3840 = 2*(2237 - 317): the model predicts occupation 1
# for everybody, so the optimizer is stuck in a flat region.
#
# Is the objective function globally concave? No. The same objective gives
# very different estimates with different starting values. But the MLE
# objective is globally concave for this model, this is why it is good to
# start GMM from the MLE estimates.

#---------------------------------------------------
# Question 3: Simulate data and recover the parameters
#---------------------------------------------------
# N = 100,000, J = 4, K = 3.
# True β:           [1.0 -2.0 0.0; -1.0 0.5 -0.5; 0.5 0.3 2.0]
# sim_logit:        [1.014 -1.914 0.023; -1.014 0.476 -0.509; 0.498 0.295 2.002]
# sim_logit_gumbel: [0.997 -1.994 0.004; -0.998 0.522 -0.474; 0.513 0.289 2.011]
# The estimates are very close to the true values with the two methods,
# so the simulation works.

#---------------------------------------------------
# Question 4: 
#---------------------------------------------------

#---------------------------------------------------
# Question 5: Multinomial logit by SMM
#---------------------------------------------------
# I use D = 1000 draws, MLE starting values and Nelder-Mead (the SMM
# objective is not smooth, so we cannot use LBFGS).
# SMM estimates have the same signs and are close to the MLE and GMM
# estimates (max difference with MLE is 0.47). Nelder-Mead did not converge
# after 1000 iterations, the objective only went from 1701.6 to 1700.8,
# so the estimates stay close to the starting values.

#---------------------------------------------------
# Question 6: all the code is in functions, main() runs everything.
# Question 7: the unit tests are in PS7_Dohou_tests.jl, all 61 tests pass.
#---------------------------------------------------
