#---------------------------------------------------
# ECON 6343: Econometrics III — Problem Set 5 (Bus Engine Replacement, Rust 1987)
# Caleb Dohou
#
# Script file: load packages and source file, then run everything.
#---------------------------------------------------

using Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM

cd(@__DIR__)

# Read in function to create state transitions for dynamic model (provided)
include("create_grids.jl")
# Read in the functions
include("PS5_Dohou_source.jl")

# Run the main source function: static estimation (Q1-Q2) then dynamic (Q3).
# The semicolon stops the REPL from re-displaying the returned objects.
θ̂_static, result_dynamic = main();

#---------------------------------------------------
# Writeup
#---------------------------------------------------
# Question 1
#   I reshape the data from wide to long with the code of the starter.
#   df_long has 20,000 rows (1,000 buses x 20 periods).
#
# Question 2 (static model, β = 0)
#   I estimate a binary logit of Y on Odometer and Branded with GLM:
#       θ0 = 1.926,  θ1 = -0.148,  θ2 = 1.059
#   θ1 is negative: when the mileage is high, Zurcher has less utility to
#   keep the engine, so he replaces it more. θ2 is positive: branded buses
#   are kept more.
#
# Question 3 (dynamic model, β = 0.9)
#   I solve the model by backward recursion on all the grid of states
#   (compute_future_value!), then I maximize the log likelihood on the
#   observed states (log_likelihood_dynamic) with Optim, starting from the
#   static estimates. It converges in 20 iterations (about 45 seconds):
#       θ0 = 1.946,  θ1 = -0.164,  θ2 = 1.074,  log likelihood = -8765.84
#   The signs are the same as in the static model, but the coefficients are
#   a little bigger in absolute value. The static model does not see that
#   replacing today also reduces the mileage cost of the future, so it puts
#   this effect inside the flow utility. The dynamic model has the future
#   value term, so the cost of mileage is estimated bigger.
#
# Question 4
#   The unit tests are in PS5_Dohou_tests.jl. This file reads the source
#   file first, then runs the tests.
