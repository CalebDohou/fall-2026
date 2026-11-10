#---------------------------------------------------
# ECON 6343: Econometrics III — Problem Set 8 (Factor models)
# Caleb Dohou
#
# Source file: all the functions. The script file loads the packages,
# includes lgwt.jl and this file, and then runs main().
#---------------------------------------------------

# The six ASVAB scores. We always select them by name, never by position.
const ASVAB_COLS = [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]

#==================================================================================
# Question 1: Load data and estimate base regression
==================================================================================#

"""
    load_data(url::String)

Load the NLSY dataset from the given URL and return it as a DataFrame.
"""
function load_data(url::String)
    return CSV.read(HTTP.get(url).body, DataFrame)
end

"""
    estimate_base_regression(df::DataFrame)

Baseline wage regression without ASVAB scores:
logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr
"""
function estimate_base_regression(df::DataFrame)
    return lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), df)
end

#==================================================================================
# Question 2: Compute correlations among ASVAB scores
==================================================================================#

"""
    compute_asvab_correlations(df::DataFrame)

Correlation matrix (6×6) of the six ASVAB scores, returned as a DataFrame.
The first column gives the row names; the other columns are named after the tests.
"""
function compute_asvab_correlations(df::DataFrame)
    asvabMat = Matrix{Float64}(df[:, ASVAB_COLS])
    correlation = cor(asvabMat)
    cordf = DataFrame(correlation, ASVAB_COLS)
    insertcols!(cordf, 1, :test => String.(ASVAB_COLS))
    return cordf
end

#==================================================================================
# Question 3: Regression with all ASVAB scores
==================================================================================#

"""
    estimate_full_regression(df::DataFrame)

Wage regression with all six ASVAB scores added to the base model.
"""
function estimate_full_regression(df::DataFrame)
    return lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr +
                       asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK), df)
end

#==================================================================================
# Question 4: PCA regression
==================================================================================#

"""
    asvab_matrix(df::DataFrame)

The six ASVAB scores as a J×N matrix (tests in rows, people in columns),
which is the shape MultivariateStats wants.
"""
function asvab_matrix(df::DataFrame)
    return Matrix{Float64}(df[:, ASVAB_COLS])'
end

"""
    estimate_pca_regression(df::DataFrame)

Fit PCA on the six ASVAB scores, take the first principal component and add it
as one regressor to the base model. Returns (regression, first PC as a vector).
The sign of a principal component is arbitrary.
"""
function estimate_pca_regression(df::DataFrame)
    asvabMat = asvab_matrix(df)                          # J×N
    M = fit(PCA, asvabMat; maxoutdim=1)
    asvabPCA = MultivariateStats.transform(M, asvabMat)  # 1×N
    df2 = @transform(df, :asvabPCA = vec(asvabPCA))      # reshape 1×N -> N vector
    est = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabPCA), df2)
    return est, vec(asvabPCA)
end

#==================================================================================
# Question 5: Factor Analysis regression
==================================================================================#

"""
    estimate_factor_regression(df::DataFrame)

Same as Question 4 but with FactorAnalysis instead of PCA.
Only the six ASVAB columns (selected by name) go into the factor analysis,
so a PCA column added to df earlier cannot slip in.
Returns (regression, first factor as a vector).
"""
function estimate_factor_regression(df::DataFrame)
    asvabMat = asvab_matrix(df)                                # J×N
    M = fit(FactorAnalysis, asvabMat; maxoutdim=1)
    asvabFactor = MultivariateStats.transform(M, asvabMat)     # 1×N
    df2 = @transform(df, :asvabFactor = vec(asvabFactor))
    est = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabFactor), df2)
    return est, vec(asvabFactor)
end

#==================================================================================
# Question 6: Full factor model with MLE
==================================================================================#

"""
    prepare_factor_matrices(df::DataFrame)

# Returns
- `X`: wage equation covariates (N×7: black, hispanic, female, schoolt, gradHS, grad4yr, constant)
- `y`: log wage (N vector)
- `Xfac`: measurement equation covariates (N×4: black, hispanic, female, constant)
- `asvabs`: the six ASVAB scores (N×6)
"""
function prepare_factor_matrices(df::DataFrame)
    N = nrow(df)
    X      = hcat(Matrix{Float64}(df[:, [:black, :hispanic, :female, :schoolt, :gradHS, :grad4yr]]), ones(N))
    y      = Vector{Float64}(df.logwage)
    Xfac   = hcat(Matrix{Float64}(df[:, [:black, :hispanic, :female]]), ones(N))
    asvabs = Matrix{Float64}(df[:, ASVAB_COLS])
    return X, y, Xfac, asvabs
end

"""
    unpack_params(θ, K, L, J)

Split θ into (α, β, γ, δ, σ). The ordering of θ is:
- α: L×J matrix, stored column by column (θ[1:L] is test 1, θ[L+1:2L] is test 2, ...)
- β: K wage coefficients
- γ: J factor loadings in the measurement equations
- δ: factor loading in the wage equation (scalar)
- σ: J+1 standard deviations (J tests, then wage)
Total length: L*J + K + J + 1 + J + 1.
"""
function unpack_params(θ::AbstractVector, K::Integer, L::Integer, J::Integer)
    α = reshape(θ[1:J*L], L, J)
    β = θ[J*L+1 : J*L+K]
    γ = θ[J*L+K+1 : J*L+K+J]
    δ = θ[J*L+K+J+1]
    σ = θ[J*L+K+J+2 : end]
    return α, β, γ, δ, σ
end

"""
    factor_model(θ, X, Xfac, Meas, y, R)

Negative log-likelihood of the factor model.

Measurement equations: asvab_j = Xfac*α_j + γ_j*ξ + ε_j,  ε_j ~ N(0, σ_j²)
Wage equation:         logwage = X*β + δ*ξ + ε,          ε ~ N(0, σ_w²)
Latent factor:         ξ ~ N(0,1)

L_i = ∫ [∏_j φ((M_ij - Xfac_i α_j - γ_j ξ)/σ_j)/σ_j] × φ((y_i - X_i β - δ ξ)/σ_w)/σ_w × φ(ξ) dξ

The integral is approximated with R Gauss-Legendre points on [-5, 5]
(lgwt.jl), which covers essentially all of the N(0,1) mass. Use R ≥ 25:
with R = 9 the log-likelihood is off by about 200 at the optimum, while
R = 25 and R = 50 agree to three decimals.
The σ's enter as sqrt(σ²) = |σ|, so the sign of σ does not matter.
"""
function factor_model(θ::Vector{T}, X::Matrix, Xfac::Matrix, Meas::Matrix,
                      y::Vector, R::Integer) where T<:Real
    K = size(X, 2)      # covariates in wage equation
    L = size(Xfac, 2)   # covariates in measurement equations
    J = size(Meas, 2)   # number of ASVAB tests
    N = length(y)

    α, β, γ, δ, σ = unpack_params(θ, K, L, J)

    ξ, ω = lgwt(R, -5, 5)

    like  = zeros(T, N)
    Mlike = zeros(T, N, J)

    # These do not depend on ξ, so compute them once
    Xα = Xfac * α      # N×J
    Xβ = X * β         # N

    for r in 1:R
        # measurement equations
        for j in 1:J
            Mres = Meas[:, j] .- Xα[:, j] .- γ[j] * ξ[r]
            sdj  = sqrt(σ[j]^2)
            Mlike[:, j] = (1 / sdj) .* pdf.(Normal(0, 1), Mres ./ sdj)
        end

        # wage equation
        Yres  = y .- Xβ .- δ * ξ[r]
        sdy   = sqrt(σ[end]^2)
        Ylike = (1 / sdy) .* pdf.(Normal(0, 1), Yres ./ sdy)

        # add this node's contribution to the integral
        like .+= ω[r] .* vec(prod(Mlike; dims = 2)) .* Ylike .* pdf(Normal(0, 1), ξ[r])
    end

    return -sum(log.(like))
end

"""
    starting_values(X, y, Xfac, asvabs)

Starting values for the MLE:
- α: OLS of each ASVAB on Xfac
- β: OLS of log wage on X
- γ, δ: 0.5 (all-zero loadings are a stationary point, because the
  likelihood is symmetric in the sign of the factor)
- σ: 0.5
"""
function starting_values(X::Matrix, y::Vector, Xfac::Matrix, asvabs::Matrix)
    J = size(asvabs, 2)
    α0 = Xfac \ asvabs        # L×J, each column is one OLS regression
    β0 = X \ y
    γ0 = fill(0.5, J)
    δ0 = 0.5
    σ0 = fill(0.5, J + 1)
    return vcat(vec(α0), β0, γ0, δ0, σ0)
end

"""
    run_estimation(df::DataFrame, start_vals::Vector; R=25, show_trace=true)

Estimate the factor model by MLE with Newton's method (ForwardDiff Hessian,
BackTracking line search). Standard errors come from the inverse Hessian
of the negative log-likelihood.

# Returns
- `θ̂`: estimates
- `se`: standard errors
- `loglike`: log-likelihood at the optimum (a negative number)
- `converged`: whether Optim reports convergence
"""
function run_estimation(df::DataFrame, start_vals::Vector; R::Integer = 25, show_trace::Bool = true)
    X, y, Xfac, asvabs = prepare_factor_matrices(df)

    td = TwiceDifferentiable(th -> factor_model(th, X, Xfac, asvabs, y, R),
                             start_vals; autodiff = Optim.ADTypes.AutoForwardDiff())

    result = optimize(td, start_vals, Newton(linesearch = BackTracking()),
                      Optim.Options(g_tol = 1e-5, iterations = 1_000,
                                    show_trace = show_trace, show_every = 1))

    θ̂  = Optim.minimizer(result)
    H  = Optim.hessian!(td, θ̂)
    se = sqrt.(diag(inv(H)))

    return θ̂, se, -Optim.minimum(result), Optim.converged(result)
end

"""
    format_results(θ::Vector, se::Vector, loglike::Real, asvabs::Matrix)

Put the estimates in a DataFrame with columns: equation, variable,
coefficient, std_error, t_stat. The rows follow the ordering of θ.
The σ's are reported as absolute values (their sign is not identified).
"""
function format_results(θ::Vector, se::Vector, loglike::Real, asvabs::Matrix)
    J = size(asvabs, 2)
    tests  = String.(ASVAB_COLS)[1:J]
    xfac_names = ["black", "hispanic", "female", "constant"]
    x_names    = ["black", "hispanic", "female", "schoolt", "gradHS", "grad4yr", "constant"]

    equation = String[]
    variable = String[]

    # α: measurement equations, one block per test
    for j in 1:J, v in xfac_names
        push!(equation, tests[j]); push!(variable, v)
    end
    # β: wage equation
    for v in x_names
        push!(equation, "logwage"); push!(variable, v)
    end
    # γ: factor loadings in measurement equations
    for j in 1:J
        push!(equation, tests[j]); push!(variable, "factor loading (γ)")
    end
    # δ
    push!(equation, "logwage"); push!(variable, "factor loading (δ)")
    # σ
    for j in 1:J
        push!(equation, tests[j]); push!(variable, "σ")
    end
    push!(equation, "logwage"); push!(variable, "σ")

    coef = copy(θ)
    coef[end-J:end] = abs.(coef[end-J:end])

    out = DataFrame(equation = equation, variable = variable,
                    coefficient = coef, std_error = se, t_stat = coef ./ se)
    println("Log-likelihood: ", round(loglike, digits = 3))
    return out
end

#==================================================================================
# Main execution function
==================================================================================#

"""
    main()

Run all questions of Problem Set 8 and return the results in a NamedTuple.
"""
function main()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS8-factor/nlsy.csv"

    println("="^80)
    println("Problem Set 8: Factor Models and Dimension Reduction")
    println("="^80)

    df = load_data(url)
    println("Data loaded. Dimensions: ", size(df))

    # Question 1
    println("\n" * "="^80)
    println("Question 1: Base Regression (without ASVAB)")
    println("="^80)
    est_base = estimate_base_regression(df)
    display(coeftable(est_base))

    # Question 2
    println("\n" * "="^80)
    println("Question 2: ASVAB Correlations")
    println("="^80)
    cordf = compute_asvab_correlations(df)
    show(stdout, cordf; allcols = true)
    println()

    # Question 3
    println("\n" * "="^80)
    println("Question 3: Full Regression (with all ASVAB)")
    println("="^80)
    est_full = estimate_full_regression(df)
    display(coeftable(est_full))

    # Question 4
    println("\n" * "="^80)
    println("Question 4: PCA Regression")
    println("="^80)
    est_pca, pc1 = estimate_pca_regression(df)
    display(coeftable(est_pca))
    println("Correlation of PC1 with the average ASVAB score: ",
            round(cor(pc1, vec(mean(asvab_matrix(df), dims = 1))), digits = 3))

    # Question 5
    println("\n" * "="^80)
    println("Question 5: Factor Analysis Regression")
    println("="^80)
    est_fac, f1 = estimate_factor_regression(df)
    display(coeftable(est_fac))
    println("Correlation of the factor with the average ASVAB score: ",
            round(cor(f1, vec(mean(asvab_matrix(df), dims = 1))), digits = 3))
    println("Correlation of PC1 with the factor: ", round(cor(pc1, f1), digits = 3))

    # Question 6
    println("\n" * "="^80)
    println("Question 6: Full Factor Model (MLE)")
    println("="^80)
    X, y, Xfac, asvabs = prepare_factor_matrices(df)
    start_vals = starting_values(X, y, Xfac, asvabs)
    println("Number of parameters: ", length(start_vals))
    println("\nEstimating full factor model...")
    θ̂, se, loglike, converged = run_estimation(df, start_vals)
    println("Converged: ", converged)
    mle_table = format_results(θ̂, se, loglike, asvabs)
    show(stdout, mle_table; allrows = true)
    println()

    println("\n" * "="^80)
    println("Analysis complete!")
    println("="^80)

    return (df = df, est_base = est_base, cordf = cordf, est_full = est_full,
            est_pca = est_pca, pc1 = pc1, est_fac = est_fac, f1 = f1,
            θ̂ = θ̂, se = se, loglike = loglike, converged = converged,
            mle_table = mle_table)
end
