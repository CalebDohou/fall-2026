#---------------------------------------------------
# ECON 6343: Econometrics III — Problem Set 3 (GEV models)
# Caleb Dohou
#
# Source file: here are all the functions. Script and tests file include it.
#---------------------------------------------------

#---------------------------------------------------
# Load data
#---------------------------------------------------
function load_data(url)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occupation
    return df, X, Z, y
end

#---------------------------------------------------
# Make matrix of choice dummies (N x J)
#---------------------------------------------------
function choice_indicators(y, J)
    N = length(y)
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = y .== j
    end
    return bigY
end

#---------------------------------------------------
# Question 1: Multinomial logit
#---------------------------------------------------

# Compute the choice probabilities (N x J)
function mlogit_probs(θ, X, Z, J)
    # θ = [21 alphas; gamma], we put gamma in last position
    α = θ[1:end-1]
    γ = θ[end]

    K = size(X, 2)
    N = size(X, 1)

    # K x J matrix of betas, last column is 0 because of normalization
    bigα = [reshape(α, K, J-1) zeros(K)]

    T = promote_type(eltype(X), eltype(θ))
    num = zeros(T, N, J)

    # numerator: exp(X*β_j + γ*(Z_j - Z_J))
    for j = 1:J
        num[:, j] = exp.(X * bigα[:, j] .+ γ .* (Z[:, j] .- Z[:, J]))
    end

    # denominator: sum of numerators
    dem = sum(num, dims=2)

    return num ./ dem
end

# Negative log-likelihood (we minimize it)
function mlogit_with_Z(θ, X, Z, y)
    J = size(Z, 2)
    bigY = choice_indicators(y, J)
    P = mlogit_probs(θ, X, Z, J)
    return -sum(bigY .* log.(P))
end

#---------------------------------------------------
# Question 3: Nested logit
#---------------------------------------------------

# Compute the choice probabilities (N x J)
# nesting_structure = [[1, 2, 3], [4, 5, 6, 7]], alternative 8 (Other) is not in any nest
function nested_logit_probs(θ, X, Z, J, nesting_structure)
    # θ = [β_WC; β_BC; λ_WC; λ_BC; γ]
    K = size(X, 2)
    N = size(X, 1)
    β = [θ[1:K], θ[K+1:2K]]
    λ = θ[2K+1:2K+2]
    γ = θ[end]

    T = promote_type(eltype(X), eltype(θ))
    lidx = zeros(T, N, J)
    num = zeros(T, N, J)

    # exp((X*β_nest + γ*(Z_j - Z_J)) / λ_nest), for Other it is equal to 1
    for j = 1:J
        nest = findfirst(n -> j in n, nesting_structure)
        if nest === nothing
            lidx[:, j] .= one(T)
        else
            lidx[:, j] = exp.((X * β[nest] .+ γ .* (Z[:, j] .- Z[:, J])) ./ λ[nest])
        end
    end

    # numerator: lidx_j * (sum of lidx in the nest)^(λ - 1)
    for j = 1:J
        nest = findfirst(n -> j in n, nesting_structure)
        if nest === nothing
            num[:, j] = lidx[:, j]
        else
            inclusive = sum(lidx[:, nesting_structure[nest]], dims=2)
            num[:, j] = lidx[:, j] .* inclusive .^ (λ[nest] - 1)
        end
    end

    # denominator: sum of numerators
    dem = sum(num, dims=2)

    return num ./ dem
end

# Negative log-likelihood (we minimize it)
function nested_logit_with_Z(θ, X, Z, y, nesting_structure)
    J = size(Z, 2)
    bigY = choice_indicators(y, J)
    P = nested_logit_probs(θ, X, Z, J, nesting_structure)
    return -sum(bigY .* log.(P))
end

#---------------------------------------------------
# Estimation
#---------------------------------------------------

function optimize_mlogit(X, Z, y; show_trace=false)
    # starting values: 21 alphas + 1 gamma
    startvals = [2*rand(7*size(X,2)).-1; 0.1]

    result = optimize(theta -> mlogit_with_Z(theta, X, Z, y),
                     startvals, LBFGS(),
                     Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=show_trace))

    return result.minimizer
end

function optimize_nested_logit(X, Z, y, nesting_structure; show_trace=false)
    # starting values: β = 0, λ = 1, γ = 0
    # (when we use random values, optimizer can get stuck near λ = 0)
    startvals = [zeros(2*size(X,2)); 1.0; 1.0; 0.0]

    result = optimize(theta -> nested_logit_with_Z(theta, X, Z, y, nesting_structure),
                     startvals, LBFGS(),
                     Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=show_trace))

    return result.minimizer
end

#---------------------------------------------------
# Standard errors from the Hessian (finite differences)
#---------------------------------------------------

function mle_se(obj, theta_hat)
    td = TwiceDifferentiable(obj, theta_hat; autodiff = Optim.ADTypes.AutoFiniteDiff())
    H = Optim.hessian!(td, theta_hat)
    # no need of minus sign, obj is already the negative log-likelihood
    return sqrt.(diag(inv(H)))
end

#---------------------------------------------------
# Question 4: main function
#---------------------------------------------------

function allwrap()
    Random.seed!(1234)

    # Load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS3-gev/nlsw88w.csv"
    df, X, Z, y = load_data(url)

    println("Data loaded successfully!")
    println("Sample size: ", size(X, 1))
    println("Number of covariates in X: ", size(X, 2))
    println("Number of alternatives: ", length(unique(y)))

    # Question 1: multinomial logit
    println("\n=== MULTINOMIAL LOGIT RESULTS ===")
    theta_hat_mle = optimize_mlogit(X, Z, y)
    println("Estimates (β_1..β_7 for [age, white, collgrad], then γ): ", theta_hat_mle)
    println("γ̂ = ", theta_hat_mle[end])

    # Question 3: nested logit
    println("\n=== NESTED LOGIT RESULTS ===")
    nesting_structure = [[1, 2, 3], [4, 5, 6, 7]]  # WC and BC
    nlogit_theta_hat = optimize_nested_logit(X, Z, y, nesting_structure)
    println("Estimates [β_WC; β_BC; λ_WC; λ_BC; γ]: ", nlogit_theta_hat)

    # Standard errors ([estimate se] in each row)
    println("\n=== STANDARD ERRORS ===")
    se_mle = mle_se(theta -> mlogit_with_Z(theta, X, Z, y), theta_hat_mle)
    println("Multinomial logit [estimate se]:")
    println([theta_hat_mle se_mle])
    se_nlogit = mle_se(theta -> nested_logit_with_Z(theta, X, Z, y, nesting_structure), nlogit_theta_hat)
    println("Nested logit [estimate se]:")
    println([nlogit_theta_hat se_nlogit])

    return nothing
end
