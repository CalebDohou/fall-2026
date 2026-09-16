#*************************************************
# ECON 6343: Econometrics III
# Problem Set 2: script (runs the source code)
# Author: Caleb Dohou
#*************************************************

# The packages are loaded here, not in the source file.
# To install them the first time, run:
#
#   import Pkg; Pkg.add("Optim")
#   import Pkg; Pkg.add("HTTP")
#   import Pkg; Pkg.add("GLM")

using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables

include("PS2_Dohou_source.jl")


#---------------------------------------------------
# question 1
#---------------------------------------------------
# writeup for question 1:
#
# Optim only minimizes, so I minimize negf(x) = -f(x) with LBFGS().
# The argmax is x* = -7.3782 and the maximum is f(x*) = 964.3134.
# This is the global maximum: the derivative of negf changes sign only one
# time, so the answer does not depend on the starting value.


#---------------------------------------------------
# question 2
#---------------------------------------------------
# writeup for question 2:
#
# I estimate the linear probability model with Optim, using the closure
# b -> ols(b, X, y). The three methods give the same estimates:
#
#                  Optim     inv(X'X)X'y    lm()
#     intercept    0.6614     0.6614       0.6614
#     age         -0.0046    -0.0046      -0.0046
#     white        0.2260     0.2260       0.2260
#     collgrad    -0.0122    -0.0122      -0.0122
#
# The difference between Optim and the closed form is only 1.5e-10.
# White women are 22.6 percentage points more likely to be married. Among the
# slope coefficients, only white is significant (t = 10.09).


#---------------------------------------------------
# question 3
#---------------------------------------------------
# writeup for question 3:
#
# logit_like() returns the negative log-likelihood, because Optim minimizes.
# I compute log(1 + exp(Xb)) in a stable way. The direct formula with p_i can
# give NaN: when p_i rounds to 1.0, log(1 - p_i) is -Inf.
#
# Estimates: intercept 0.7466, age -0.0211, white 0.9558, collgrad -0.0560.
# The log-likelihood is -1416.95. Only white is significant.


#---------------------------------------------------
# question 4
#---------------------------------------------------
# writeup for question 4:
#
# glm() gives the same coefficients as Optim. The largest difference is
# 4.2e-7, so my logit likelihood is correct.


#---------------------------------------------------
# question 5
#---------------------------------------------------
# writeup for question 5:
#
# After cleaning, N = 2237 and there are 7 occupations, so there are
# 4 x 6 = 24 parameters. Occupation 7 is the base.
#
# Estimates (rounded to 4 decimals):
#
#                    1        2        3        4        5        6
#     intercept   0.1910  -0.1699   0.6895  -2.2675  -1.3986   0.2455
#     age        -0.0335  -0.0360  -0.0105  -0.0053  -0.0143  -0.0067
#     white       0.5964   1.3068   0.5232   1.3914  -0.0177  -0.5383
#     collgrad    0.4165  -0.4310  -1.4925  -0.9850  -1.4951  -3.7898
#
# The log-likelihood is -3605.09.
#
# I start from a vector of zeros. With g_tol = 1e-5, Optim stops because the
# objective does not change anymore, not because of the gradient (the
# gradient norm is 8.5e-5). Starting from U[0,1] and U[-1,1] gives the same
# log-likelihood and estimates within 1.4e-4, so the results are good to
# about 4 decimals.
#
# q5_checks() shows the likelihood is correct: the predicted shares equal the
# observed shares (gap 2.2e-8), and a simple loop version gives the same
# log-likelihood (gap 1.3e-11).
#
# Compared to the base category, college graduates are more likely to work in
# occupation 1 and less likely to work in the other occupations.


#---------------------------------------------------
# question 6
#---------------------------------------------------
# writeup for question 6:
#
# All the code is inside allwrap(), which I call at the bottom of this script.
# It sets the seed, so the results are the same every time.


#---------------------------------------------------
# question 7
#---------------------------------------------------
# writeup for question 7:
#
# The unit tests are in PS2_Dohou_tests.jl. The file includes the source file
# and runs 119 tests for all the functions. All tests pass.


#---------------------------------------------------
# run everything (question 6)
#---------------------------------------------------
allwrap()
