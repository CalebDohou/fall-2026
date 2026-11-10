#---------------------------------------------------
# ECON 6343: Econometrics III — Problem Set 8 (Factor models)
# Caleb Dohou
#
# Script file: load packages, the quadrature helper and the source file,
# then run everything. The writeup (answers to all questions) is in the
# comments at the bottom.
#---------------------------------------------------
using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta,
      CSV, MultivariateStats, Distributions, FreqTables, ForwardDiff, LineSearches

cd(@__DIR__)

Random.seed!(1234)

# Gauss-Legendre quadrature nodes and weights (from Problem Set 4)
include("lgwt.jl")

# Read in the functions
include("PS8_Dohou_source.jl")

# Run the main function (all questions are wrapped in main())
results = main();
nothing   # so include() does not print the whole results NamedTuple in the REPL

#---------------------------------------------------
# Question 1: Base regression
#---------------------------------------------------
# black -0.167, hispanic -0.054, female -0.155, schoolt 0.005 (not
# significant), gradHS 0.196, grad4yr 0.299.
# Black, hispanic and female earn less. Graduating from high school or
# from a 4-year college increases the wage a lot.

#---------------------------------------------------
# Question 2: Correlation of the six ASVAB scores
#---------------------------------------------------
# All the correlations are positive and high, between 0.42 and 0.80
# (AR and MK is 0.80). The six tests measure almost the same thing.

#---------------------------------------------------
# Question 3: Regression with the six ASVAB
#---------------------------------------------------
# Yes, it is a problem. The ASVAB are very correlated (multicollinearity),
# so the regression cannot separate the effect of each test. Only AR and
# MK are significant, PC has a negative sign (-0.022) which does not make
# sense, and the other tests are close to zero.

#---------------------------------------------------
# Question 4: First principal component
#---------------------------------------------------
# Coefficient on PC1 = -0.053 (se 0.006).
# The sign is arbitrary. Here PC1 has correlation -0.999 with the average
# ASVAB score, so a higher PC1 means a lower score. So higher ability
# increases the wage. The other coefficients are close to Q3.

#---------------------------------------------------
# Question 5: Factor analysis
#---------------------------------------------------
# Coefficient on the factor = 0.115 (se 0.013).
# Here the factor is positively correlated with the scores (0.986), so the
# sign is positive. The factor and PC1 are almost the same variable
# (correlation -0.991), and the other coefficients are almost the same as
# in Q4. The coefficient is different only because the scale is different.

#---------------------------------------------------
# Question 6: Full measurement system by MLE
#---------------------------------------------------
# I integrate out ξ with Gauss-Legendre quadrature on [-5, 5] with 25
# points (with 9 points the likelihood was not accurate). Starting values:
# OLS for α and β, 0.5 for γ, δ and σ. Newton converged in 7 iterations.
# Log-likelihood = -17,524.72.
#
# Wage equation: black -0.187, hispanic -0.080, female -0.136,
# schoolt -0.002, gradHS 0.107, grad4yr 0.239, δ = 0.105 (se 0.012).
# So one standard deviation more of ability increases the wage by about
# 10.5%. The education coefficients are close to Q4 and Q5.
#
# All the γ are positive and very significant (0.55 to 0.82), so all the
# tests depend on ξ. MK and AR have the biggest γ and the smallest σ, so
# they measure ability the best.

#---------------------------------------------------
# Question 7: the unit tests are in PS8_Dohou_tests.jl, all 113 tests pass.
#---------------------------------------------------
