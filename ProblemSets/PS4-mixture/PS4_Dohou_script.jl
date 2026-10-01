#---------------------------------------------------
# ECON 6343: Econometrics III — Problem Set 4 (Mixed Logit Models)
# Caleb Dohou
#
# Script file: load packages and source file, then run everything.
#---------------------------------------------------

using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, Distributions, ForwardDiff

cd(@__DIR__)

# Quadrature function (lgwt.jl must be in the same folder)
include("lgwt.jl")
# Read in the functions
include("PS4_Dohou_source.jl")

#---------------------------------------------------
# Question 2: Does γ̂ make more sense now than in PS3?
#---------------------------------------------------
# Yes. In PS3 we had γ̂ = -0.094 (se 0.379), negative and not significant.
# Now γ̂ = 1.307 (se 0.125), positive and significant (t ≈ 10.5).
# It makes more sense: when the expected wage of an occupation is higher,
# people have more utility to choose this occupation.

#---------------------------------------------------
# Question 3: Quadrature and Monte Carlo practice
#---------------------------------------------------
# (a) ∫φ(x)dx = 1.0045 and ∫xφ(x)dx ≈ 0, so it is verified.
#
# (b) True variance is 4. With 7 points we get 3.266 and with 10 points 4.039.
#     7 points is not enough (error of 18%), but 10 points is already
#     very close (error of 1%).
#
# (c) With D = 1,000,000 we get 3.996, -0.001 and 1.0002, very close to 4, 0 and 1.
#     With D = 1,000 we get 3.844, 0.042 and 1.099, so it is much less precise.
#     Monte Carlo needs a lot of draws to approximate well the integral.

#---------------------------------------------------
# Questions 4 and 5: Mixed logit
#---------------------------------------------------
# The code is in the source file. We do not run it (too slow), the two calls
# are commented in allwrap().

#---------------------------------------------------
# Question 6: call the main function
#---------------------------------------------------
allwrap()
