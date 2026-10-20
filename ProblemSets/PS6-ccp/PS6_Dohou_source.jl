#---------------------------------------------------
# ECON 6343: Econometrics III — Problem Set 6 (Rust 1987, CCP estimation)
# Caleb Dohou
#
# Source file: all the functions. The script file runs @time main().
#---------------------------------------------------

#========================================
QUESTION 1: Data Loading and Reshaping
========================================#

"""
    load_and_reshape_data(url::String)

Read busdata.csv (from PS5) and reshape it to long format.
Returns df_long (sorted by bus and time), and Xstate (N × T), Zstate and
Branded in wide format for Q3.
"""
function load_and_reshape_data(url::String)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    T  = 20

    # Create bus id variable
    df = @transform(df, :bus_id = 1:size(df,1))

    # Reshape Y1-Y20 to long
    dfy = df[:, vcat(:bus_id, [Symbol("Y$t") for t in 1:T], :RouteUsage, :Branded)]
    dfy_long = DataFrames.stack(dfy, Not([:bus_id, :RouteUsage, :Branded]))
    rename!(dfy_long, :value => :Y)
    # stack() puts all buses for Y1 first, then Y2, etc.
    dfy_long = @transform(dfy_long, :time = kron(collect(1:T), ones(size(df,1))))
    select!(dfy_long, Not(:variable))

    # Reshape Odo1-Odo20 to long
    dfx = df[:, vcat(:bus_id, [Symbol("Odo$t") for t in 1:T])]
    dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
    rename!(dfx_long, :value => :Odometer)
    dfx_long = @transform(dfx_long, :time = kron(collect(1:T), ones(size(df,1))))
    select!(dfx_long, Not(:variable))

    # Join the two and sort by bus then time (needed for FVT1'[:] in Q3)
    df_long = innerjoin(dfy_long, dfx_long, on = [:bus_id, :time])
    sort!(df_long, [:bus_id, :time])

    # State variables in wide format, like in PS5
    Xstate  = Matrix(df[:, [Symbol("Xst$t") for t in 1:T]])
    Zstate  = Vector(df[:, :Zst])
    Branded = Vector(df[:, :Branded])

    return df_long, Xstate, Zstate, Branded
end


#========================================
QUESTION 2: Flexible Logit Estimation
========================================#

"""
    estimate_flexible_logit(df::DataFrame)

Flexible logit with all the interactions of Odometer, Odometer², RouteUsage,
RouteUsage², Branded, time and time² (128 terms). The squares are inside
@formula so predict() updates them in Q3. GLM drops 37 terms (also the
intercept) because the columns are very big (up to 1e8).
"""
function estimate_flexible_logit(df::DataFrame)
    flex_logit = glm(@formula(Y ~ Odometer * Odometer^2 * RouteUsage * RouteUsage^2 *
                                  Branded * time * time^2),
                     df, Binomial(), LogitLink())
    return flex_logit
end


#========================================
QUESTION 3: CCP-Based Estimation
========================================#

"""
    construct_state_space(xbin::Int, zbin::Int, xval::Vector, zval::Vector)

Data frame with all the (mileage, route usage) states, in the same order as
the rows of xtran. Branded and time are 0 for now.
"""
function construct_state_space(xbin::Int, zbin::Int, xval::Vector, zval::Vector)
    state_df = DataFrame(
        Odometer   = kron(ones(zbin), xval),
        RouteUsage = kron(zval, ones(xbin)),
        Branded    = zeros(xbin*zbin),
        time       = zeros(xbin*zbin)
    )
    return state_df
end


"""
    compute_future_values(state_df, flex_logit, xtran, T, β)

FV = -β log p0 for all the states. Size is (rows of xtran, 2, T+1).
FV[:,:,T+1] stays 0 because there is no future after the last period.
"""
function compute_future_values(state_df::DataFrame,
                               flex_logit,
                               xtran::Matrix,
                               T::Int,
                               β::Float64)

    FV = zeros(size(xtran,1), 2, T+1)

    for t in 2:T
        for b in 0:1
            # Put the time and brand in the state data frame
            state_df.time    .= t
            state_df.Branded .= b
            # p0 = probability to replace = 1 - Pr(Y = 1)
            p0 = 1 .- convert(Vector{Float64}, predict(flex_logit, state_df))
            # For some states p0 = 0, so we use eps() to avoid log(0)
            FV[:, b+1, t] .= -β .* log.(max.(p0, eps()))
        end
    end

    return FV
end


"""
    compute_fvt1(df_long, FV, xtran, Xstate, Zstate, xbin, B)

Put the future value on the data:
FVT1[i,t] = (xtran[row1,:] - xtran[row0,:])' * FV[row0:row0+xbin-1, B[i]+1, t+1]
row0 is the replace state (mileage 0) and row1 is the actual state.
Returns a long vector (bus, then time), same order as df_long.
"""
function compute_fvt1(df_long::DataFrame,
                      FV::Array{Float64,3},
                      xtran::Matrix,
                      Xstate::Matrix,
                      Zstate::Vector,
                      xbin::Int,
                      B::Vector)

    N = length(unique(df_long.bus_id))
    T = size(Xstate, 2)

    FVT1 = zeros(N, T)
    for i in 1:N
        row0 = (Zstate[i]-1)*xbin + 1
        for t in 1:T
            row1 = row0 + Xstate[i,t] - 1
            FVT1[i,t] = (xtran[row1,:] .- xtran[row0,:])' * FV[row0:row0+xbin-1, B[i]+1, t+1]
        end
    end

    # Long format: bus 1 (t = 1..T), bus 2 (t = 1..T), ...
    fvt1_long = FVT1'[:]

    return fvt1_long
end


"""
    estimate_structural_params(df_long::DataFrame, fvt1::Vector)

Logit of Y on Odometer and Branded, with the future value as offset
(its coefficient is fixed to 1).
"""
function estimate_structural_params(df_long::DataFrame, fvt1::Vector)
    df_long = @transform(df_long, :fv = fvt1)

    theta_hat = glm(@formula(Y ~ Odometer + Branded),
                    df_long,
                    Binomial(),
                    LogitLink(),
                    offset = df_long.fv)

    return theta_hat
end


#========================================
QUESTION 3(d): Own logit with an offset
========================================#

"""
    logit_offset_loglike(θ, X, y, offset)

Log likelihood of the logit with index X*θ + offset.
I write log(1 + exp(η)) in a stable way to avoid overflow.
"""
function logit_offset_loglike(θ::Vector, X::Matrix, y::Vector, offset::Vector)
    η = X*θ .+ offset
    return sum(y .* η .- (max.(η, 0) .+ log1p.(exp.(-abs.(η)))))
end


"""
    estimate_logit_offset(X, y, offset; θ0 = zeros(size(X,2)))

Maximize the log likelihood with Optim (LBFGS and the gradient).
Standard errors from the inverse of X'WX, with W = p(1-p).
Returns θ̂, se and the log likelihood.
"""
function estimate_logit_offset(X::Matrix, y::Vector, offset::Vector;
                               θ0::Vector = zeros(size(X,2)))
    Λ(η) = 1 ./ (1 .+ exp.(-η))

    f(θ) = -logit_offset_loglike(θ, X, y, offset)
    function g!(G, θ)
        p = Λ(X*θ .+ offset)
        G .= -X' * (y .- p)
    end

    res = optimize(f, g!, θ0, LBFGS(),
                   Optim.Options(g_tol = 1e-8, iterations = 1_000))
    θ̂ = Optim.minimizer(res)

    p  = Λ(X*θ̂ .+ offset)
    H  = X' * (X .* (p .* (1 .- p)))
    se = sqrt.(diag(inv(H)))

    return θ̂, se, -Optim.minimum(res)
end


#========================================
MAIN WRAPPER FUNCTION
========================================#

"""
    main()

Run all the steps (Q1 to Q3) and print the results.
Returns the GLM estimates and my own estimates.
"""
function main()
    println("="^60)
    println("Rust (1987) Bus Engine Replacement Model - CCP Estimation")
    println("="^60)

    β = 0.9  # Discount factor

    # Step 1: Load and reshape data
    println("\nStep 1: Loading and reshaping data...")
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS5-ddc/busdata.csv"
    df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
    T = size(Xstate, 2)

    # Step 2: Estimate flexible logit
    println("\nStep 2: Estimating flexible logit...")
    flex_logit = estimate_flexible_logit(df_long)
    # The dropped terms have a coefficient equal to 0
    ndrop = sum(coef(flex_logit) .== 0)
    println("Flexible logit: ", length(coef(flex_logit)), " terms, ",
            ndrop, " dropped by GLM")

    # Step 3a: State transition matrices (same as PS5)
    println("\nStep 3a: Constructing state transition matrices...")
    zval, zbin, xval, xbin, xtran = create_grids()

    # Step 3b: State space
    println("\nStep 3b: Constructing state space...")
    state_df = construct_state_space(xbin, zbin, xval, zval)

    # Step 3c: Future values for all the states
    println("\nStep 3c: Computing future values...")
    FV = compute_future_values(state_df, flex_logit, xtran, T, β)

    # Step 3d: Future values on the data
    println("\nStep 3d: Mapping future values to data...")
    fvt1 = compute_fvt1(df_long, FV, xtran, Xstate, Zstate, xbin, Branded)

    # Step 3e: Structural parameters with GLM and offset
    println("\nStep 3e: Estimating structural parameters...")
    theta_hat_ccp_glm = estimate_structural_params(df_long, fvt1)

    # Q3(d): my own logit with offset
    X = hcat(ones(nrow(df_long)), df_long.Odometer, df_long.Branded)
    y = Float64.(df_long.Y)
    θ̂_own, se_own, ll_own = estimate_logit_offset(X, y, fvt1)

    # Print results
    println("\n" * "="^60)
    println("RESULTS")
    println("="^60)
    println("\nCCP estimates with GLM (offset = future value):")
    show(stdout, MIME("text/plain"), coeftable(theta_hat_ccp_glm)); println()
    println("\nCCP estimates with our own logit with offset:")
    println(DataFrame(term = ["θ0 (Intercept)", "θ1 (Odometer)", "θ2 (Branded)"],
                      estimate = θ̂_own, std_error = se_own))
    println("Log likelihood: ", round(ll_own, digits = 2))

    return theta_hat_ccp_glm, θ̂_own
end
