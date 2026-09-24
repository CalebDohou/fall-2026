#---------------------------------------------------
# ECON 6343: Econometrics III — Problem Set 3 (GEV models)
# Caleb Dohou
#
# Script file: load packages and source file, then run everything.
#---------------------------------------------------

using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables

cd(@__DIR__)

# Read in the functions
include("PS3_Dohou_source.jl")

#---------------------------------------------------
# Question 2: Interpretation of γ̂
#---------------------------------------------------
# γ is the effect of expected log wage on the utility of choosing an occupation.
# It is same for all occupations. Since Z is in log, when expected wage of
# occupation j increase by 1%, the utility of choosing j change by about γ/100.
#
# In the MNL we get γ̂ = -0.094 (se 0.379). The sign is negative, which is
# strange because higher wage should make the occupation more attractive.
# But it is small and not significant (t ≈ -0.25), so we cannot say that
# expected wage affect the choice of occupation once we control for X.
#
# Note: in the nested logit γ̂ = 0.672, but λ̂_WC = -0.284 and λ̂_BC = -0.247 are
# outside (0,1], so this model is not consistent with utility maximization.
# Also the wage effect inside a nest is γ/λ, which is negative, so this γ̂ is
# not comparable with the MNL one.

#---------------------------------------------------
# Question 4: call the main function
#---------------------------------------------------
allwrap()
