#---------------------------------------------------
# ECON 6343: Econometrics III — Problem Set 5 (Bus Engine Replacement, Rust 1987)
# Caleb Dohou
#
# Source file: all the functions are here.
# The script file (PS5_Dohou_script.jl) loads the packages, includes
# create_grids.jl and this file, and then calls main().
#---------------------------------------------------

################################################################################
# PART 1: LOAD THE DATA (Questions 1 and 3a)
################################################################################

"""
    load_static_data()

Question 1. Read busdataBeta0.csv and put it in long format (one row for
each bus and each period). Returns df_long.
"""
function load_static_data()
    # Load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    # Create bus id variable
    df = @transform(df, :bus_id = 1:size(df,1))

    #---------------------------------------------------
    # Reshape from wide to long. We do it two times because
    # stack() works with one variable at a time.
    #---------------------------------------------------

    # First the decision variable (Y1-Y20)
    dfy = @select(df, :bus_id, :Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10,
                      :Y11, :Y12, :Y13, :Y14, :Y15, :Y16, :Y17, :Y18, :Y19, :Y20,
                      :RouteUsage, :Branded)
    dfy_long = DataFrames.stack(dfy, Not([:bus_id, :RouteUsage, :Branded]))
    rename!(dfy_long, :value => :Y)
    dfy_long = @transform(dfy_long, :time = kron(collect(1:20), ones(size(df,1))))
    select!(dfy_long, Not(:variable))

    # Then the odometer variable (Odo1-Odo20)
    dfx = @select(df, :bus_id, :Odo1, :Odo2, :Odo3, :Odo4, :Odo5, :Odo6, :Odo7,
                      :Odo8, :Odo9, :Odo10, :Odo11, :Odo12, :Odo13, :Odo14, :Odo15,
                      :Odo16, :Odo17, :Odo18, :Odo19, :Odo20)
    dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
    rename!(dfx_long, :value => :Odometer)
    dfx_long = @transform(dfx_long, :time = kron(collect(1:20), ones(size(df,1))))
    select!(dfx_long, Not(:variable))

    # Join the two long dataframes
    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id, :time])
    sort!(df_long, [:bus_id, :time])

    return df_long
end

"""
    load_dynamic_data()

Question 3a. Read busdata.csv and keep it in wide format. Y, Odo and Xst
become matrices N×T, Zst and Branded are vectors. Also create the grids
(question 3b) and put everything in a named tuple.
"""
function load_dynamic_data()
    # Load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS5-ddc/busdata.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    # Take the matrices and vectors
    Y      = Matrix(df[:, r"^Y\d+"])    # decision (N × T): 1 = run the bus, 0 = replace
    X      = Matrix(df[:, r"^Odo\d+"])  # odometer (N × T), in 10,000 miles
    Xstate = Matrix(df[:, r"^Xst"])     # bin of the odometer (N × T)
    Zstate = Vector(df[:, :Zst])        # bin of the route usage (N)
    B      = Vector(df[:, :Branded])    # branded dummy (N)

    N, T = size(Y)

    # Question 3b: grids and transition matrix (code given by the professor)
    zval, zbin, xval, xbin, xtran = create_grids()

    return (
        # Data
        Y = Y,
        X = X,
        B = B,
        Xstate = Xstate,
        Zstate = Zstate,
        # Dimensions
        N = N,
        T = T,
        # Grids
        xval = xval,
        xbin = xbin,
        zval = zval,
        zbin = zbin,
        xtran = xtran,
        # Discount factor
        β = 0.9
    )
end

################################################################################
# PART 2: STATIC MODEL (Question 2)
################################################################################

"""
    estimate_static_model(df_long)

Question 2. Zurcher is myopic (β = 0), so the model is a simple binary logit
of Y on Odometer and Branded. I use GLM.
"""
function estimate_static_model(df_long)
    θ̂_static = glm(@formula(Y ~ Odometer + Branded), df_long, Binomial(), LogitLink())
    return θ̂_static
end

################################################################################
# PART 3: DYNAMIC MODEL - FUTURE VALUE (Question 3c)
################################################################################

#=
    compute_future_value!(FV, θ, d)

Question 3c. Solve the model by backward recursion for one value of θ.

FV has size (zbin*xbin, 2, T+1). In FV[row, b+1, t] we store
    β·log(exp(v0t) + exp(v1t))
for the state row = x + (z-1)*xbin, brand b and period t. The last slice
FV[:, :, T+1] stays at zero because V_{T+1} = 0.

On the grid:
    v1t = θ0 + θ1·xval[x] + θ2·b + xtran[row,:]' · FV[block of z, b+1, t+1]
    v0t =                        xtran[1+(z-1)*xbin,:]' · FV[block of z, b+1, t+1]
The block of z is the rows (z-1)*xbin+1 : z*xbin (all the mileage bins for
this z). For v0 the mileage goes back to zero, so we use the first row of
the block.
=#
@views @inbounds function compute_future_value!(FV, θ, d)
    # Loop backward on time (FV[:,:,T+1] is already zero)
    for t in d.T:-1:1
        # Loop on brand
        for b in 0:1
            # Loop on route usage
            for z in 1:d.zbin
                # Future values of next period for all the mileage bins of this z
                FVnext = FV[(z-1)*d.xbin+1:z*d.xbin, b+1, t+1]
                # Future value if we replace (mileage goes back to 0).
                # It is the same for all x in this block, so I compute it one time.
                row0 = 1 + (z-1)*d.xbin
                v0 = d.xtran[row0, :] ⋅ FVnext
                # Loop on mileage
                for x in 1:d.xbin
                    # Row of the transition matrix for the state (x, z)
                    row = x + (z-1)*d.xbin
                    # Value of running the bus: flow utility + expected future value
                    v1 = θ[1] + θ[2]*d.xval[x] + θ[3]*b + d.xtran[row, :] ⋅ FVnext
                    # Save β·log(exp(v0) + exp(v1)); it is the future value for period t-1
                    FV[row, b+1, t] = d.β * log(exp(v1) + exp(v0))
                end
            end
        end
    end
    return nothing
end

################################################################################
# PART 4: DYNAMIC MODEL - LOG LIKELIHOOD (Questions 3d, 3e, 3f)
################################################################################

#=
    log_likelihood_dynamic(θ, d)

Questions 3d-3f. Negative log likelihood of the dynamic model. θ is the
first argument so the function can go directly in Optim.

Step 1: solve the model for this θ (compute_future_value!).
Step 2: only on the observed states, for each bus i and period t,
    v1 - v0 = θ0 + θ1·X[i,t] + θ2·B[i]
              + β·(xtran[row1,:] - xtran[row0,:])' · FV[row0:row0+xbin-1, B[i]+1, t+1]
with row0 = 1 + (Zst[i]-1)*xbin (replaced) and row1 = Xst[i,t] + (Zst[i]-1)*xbin
(not replaced). Then
    P1 = exp(v1-v0) / (1 + exp(v1-v0)),  P0 = 1 - P1
and ℓ = Σ_i Σ_t [Y·log P1 + (1-Y)·log P0] = Σ_i Σ_t [Y·(v1-v0) - log(1 + exp(v1-v0))].
=#
@views @inbounds function log_likelihood_dynamic(θ, d)
    # Step 1: future values for all the states
    FV = zeros(d.zbin * d.xbin, 2, d.T + 1)
    compute_future_value!(FV, θ, d)

    # Step 2: likelihood with the observed states only
    loglike = 0.0
    for i in 1:d.N
        # Row if the bus is replaced (mileage = 0); same for all t
        row0 = 1 + (d.Zstate[i] - 1) * d.xbin
        for t in 1:d.T
            # Row of the observed state (not replaced)
            row1 = d.Xstate[i,t] + (d.Zstate[i] - 1) * d.xbin

            # Flow utility part of v1 - v0, with the observed mileage and brand
            flow_diff = θ[1] + θ[2]*d.X[i,t] + θ[3]*d.B[i]

            # Future value part: difference of the two rows of xtran
            ev_diff = (d.xtran[row1, :] .- d.xtran[row0, :]) ⋅ FV[row0:row0+d.xbin-1, d.B[i]+1, t+1]

            v_diff = flow_diff + d.β * ev_diff

            # Y·log P1 + (1-Y)·log P0, written in a stable way
            loglike += d.Y[i,t] * v_diff - log(1 + exp(v_diff))
        end
    end

    # Return the NEGATIVE log likelihood because Optim minimizes
    return -loglike
end

################################################################################
# PART 5: OPTIMIZATION (Questions 3e-3h)
################################################################################

"""
    estimate_dynamic_model(d; θ_start=nothing)

Maximize the dynamic log likelihood with Optim (LBFGS). The starting point
is θ_start (the static estimates of question 2). Returns the Optim result.
"""
function estimate_dynamic_model(d; θ_start=nothing)
    println("="^70)
    println("DYNAMIC MODEL ESTIMATION (β = ", d.β, ")")
    println("="^70)

    # Starting values
    if isnothing(θ_start)
        θ_start = rand(3)
        println("\nUsing random starting values: ", θ_start)
    else
        println("\nUsing static estimates as starting values: ", θ_start)
    end

    # Objective: negative log likelihood, function of θ only
    objective = θ -> log_likelihood_dynamic(θ, d)

    # Time of one evaluation (the second call is without compilation)
    println("\nTiming one likelihood evaluation...")
    @time objective(θ_start)
    @time objective(θ_start)

    # Optimize
    println("\nOptimizing (this may take a few minutes)...")
    result = optimize(objective, θ_start, LBFGS(),
                      Optim.Options(g_tol = 1e-5,
                                    iterations = 100_000,
                                    show_trace = true,
                                    show_every = 10))

    # Print the results
    println("\n" * "="^70)
    println("DYNAMIC MODEL RESULTS")
    println("="^70)
    println("Parameter estimates:")
    println("  θ₀ (constant):     ", round(result.minimizer[1], digits=4))
    println("  θ₁ (mileage):      ", round(result.minimizer[2], digits=4))
    println("  θ₂ (brand):        ", round(result.minimizer[3], digits=4))
    println("\nLog likelihood:      ", round(-result.minimum, digits=2))
    println("Converged:           ", Optim.converged(result))
    println("Iterations:          ", Optim.iterations(result))
    println("="^70)

    return result
end

################################################################################
# MAIN FUNCTION (Question 3g)
################################################################################

"""
    main()

Run all the problem set: static model (questions 1-2), then dynamic model
(question 3) starting from the static estimates. Returns the GLM fit and
the Optim result.
"""
function main()
    println("\n" * "="^70)
    println("PROBLEM SET 5: BUS ENGINE REPLACEMENT MODEL")
    println("="^70)

    #---------------------------------------------------------------------------
    # Questions 1-2: static model
    #---------------------------------------------------------------------------
    println("\n" * "-"^70)
    println("PART 1: STATIC (MYOPIC) ESTIMATION")
    println("-"^70)

    println("\nQuestion 1: loading and reshaping data to long format...")
    df_long = load_static_data()
    println("Observations in df_long: ", nrow(df_long))
    println("First few rows:")
    println(first(df_long, 5))

    println("\nQuestion 2: static logit estimates (β = 0):")
    θ̂_static = estimate_static_model(df_long)
    show(stdout, MIME("text/plain"), coeftable(θ̂_static))   # only the table, not all the model object
    println()

    # The static coefficients are the starting values of the dynamic model
    θ_start_dynamic = coef(θ̂_static)

    #---------------------------------------------------------------------------
    # Question 3: dynamic model
    #---------------------------------------------------------------------------
    println("\n" * "-"^70)
    println("PART 2: DYNAMIC ESTIMATION")
    println("-"^70)

    println("\nQuestion 3a-3b: loading wide data and building state grids...")
    d = load_dynamic_data()
    println("Buses (N):               ", d.N)
    println("Time periods (T):        ", d.T)
    println("Mileage bins (xbin):     ", d.xbin)
    println("Route usage bins (zbin): ", d.zbin)
    println("Total grid points:       ", d.xbin * d.zbin)
    println("Discount factor (β):     ", d.β)

    println("\nQuestion 3c-3h: solving and estimating the dynamic model...")
    result_dynamic = estimate_dynamic_model(d, θ_start = θ_start_dynamic)

    #---------------------------------------------------------------------------
    # Static vs dynamic
    #---------------------------------------------------------------------------
    println("\n" * "-"^70)
    println("COMPARISON: STATIC vs DYNAMIC ESTIMATES")
    println("-"^70)
    θ̂_dyn = result_dynamic.minimizer
    println("                 Static (β=0)    Dynamic (β=0.9)")
    println("  θ₀ (constant)  ", lpad(round(θ_start_dynamic[1], digits=4), 10), "       ", lpad(round(θ̂_dyn[1], digits=4), 10))
    println("  θ₁ (mileage)   ", lpad(round(θ_start_dynamic[2], digits=4), 10), "       ", lpad(round(θ̂_dyn[2], digits=4), 10))
    println("  θ₂ (brand)     ", lpad(round(θ_start_dynamic[3], digits=4), 10), "       ", lpad(round(θ̂_dyn[3], digits=4), 10))

    println("\n" * "="^70)
    println("END OF PROBLEM SET")
    println("="^70)

    return θ̂_static, result_dynamic
end
