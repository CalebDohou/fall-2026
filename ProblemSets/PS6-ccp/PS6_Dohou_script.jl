#---------------------------------------------------
# ECON 6343: Econometrics III — Problem Set 6 (Rust 1987, CCP estimation)
# Caleb Dohou
#
# Script file: load packages and source file, then run everything.
#---------------------------------------------------

using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV

cd(@__DIR__)

# Read in function to create state transitions for dynamic model (provided)
include("create_grids.jl")
# Read in the functions
include("PS6_Dohou_source.jl")

# Run everything and time it (Question 3e).
@time theta_hat_ccp_glm, θ̂_own = main()
nothing  # to not print again the objects returned by main()

#---------------------------------------------------
# Writeup
#---------------------------------------------------
# Question 1
#   I read busdata.csv and reshape it to long format like in PS5.
#   df_long has 20,000 rows (1,000 buses x 20 periods).
#
# Question 2
#   I estimate the flexible logit with the fully interacted formula of the
#   PDF. The squared terms are inside @formula, so predict() updates them in
#   Q3. GLM drops 37 of the 128 terms (also the intercept). It is fine, like
#   the PDF says.
#
# Question 3
#   (a) xtran comes from create_grids(), same as PS5.
#   (b) I compute FV = -β log p0 for all the states, for t = 2..T and b = 0,1,
#       with p0 = 1 - Pr(Y=1) from the flexible logit. Some states give
#       p0 = 0, so I use max(p0, eps()) to avoid log(0). Then compute_fvt1()
#       maps FV to the data with (xtran[row1,:] - xtran[row0,:]).
#   (c) Estimates with GLM and the future value as offset:
#           θ0 = 1.944 (0.033),  θ1 = -0.149 (0.006),  θ2 = 1.125 (0.040)
#   (d) My own logit with offset gives the same estimates.
#   (e) Everything is in main(), called with @time. It takes about 12 seconds
#       (the first run is longer, about 2-3 minutes, because of compilation).
#   (f) The results are close to PS5 (θ0 = 1.946, θ1 = -0.164, θ2 = 1.074),
#       but here we do not solve the model inside the likelihood. It is only
#       a logit with an offset, so it is much faster. CCPs are powerful!
#
# Question 4
#   The unit tests are in PS6_Dohou_tests.jl. It reads the source file
#   first, then tests each function. All 502 tests pass.
