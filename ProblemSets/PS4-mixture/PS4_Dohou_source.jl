#---------------------------------------------------
# ECON 6343: Econometrics III — Problem Set 4 (Mixed Logit Models)
# Caleb Dohou
#
# Source file: all the functions are here.
#---------------------------------------------------


#---------------------------------------------------
# Load the data
#---------------------------------------------------
function load_data()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS4-mixture/nlsw88t.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occ_code
    return df, X, Z, y
end


#---------------------------------------------------
# Question 1: Multinomial logit
#---------------------------------------------------
function mlogit_with_Z(θ, X, Z, y)
    # θ = [21 alphas; gamma]
    α = θ[1:end-1]
    γ = θ[end]

    K = size(X, 2)  # number of variables in X
    J = length(unique(y))  # number of choices (8)
    N = length(y)   # number of observations

    # matrix of choice dummies
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = y .== j
    end

    # K x J matrix of alphas, last column is 0 (normalization)
    bigα = [reshape(α, K, J-1) zeros(K)]

    # T is needed for the automatic differentiation
    T = promote_type(eltype(X), eltype(θ))
    num = zeros(T, N, J)
    dem = zeros(T, N)

    # numerator: exp(X*alpha_j + gamma*(Z_j - Z_J))
    for j = 1:J
        num[:,j] = exp.(X * bigα[:, j] .+ γ .* (Z[:, j] .- Z[:, J]))
    end

    # denominator: sum of numerators
    dem = sum(num, dims=2)

    # probabilities
    P = num ./ dem

    # negative log-likelihood (we minimize it)
    loglike = -sum(bigY .* log.(P))

    return loglike
end


#---------------------------------------------------
# Question 3a: Quadrature practice
#---------------------------------------------------

function practice_quadrature()
    println("=== Question 3a: Quadrature Practice ===")

    # standard normal
    d = Normal(0, 1)

    # 7 nodes and weights on [-4, 4]
    nodes, weights = lgwt(7, -4, 4)

    # integral of density, should be 1
    integral_density = sum(weights .* pdf.(d, nodes))
    println("∫φ(x)dx = ", integral_density, " (should be ≈ 1)")

    # expectation, should be 0
    expectation = sum(weights .* nodes .* pdf.(d, nodes))
    println("∫xφ(x)dx = ", expectation, " (should be ≈ 0)")
end

#---------------------------------------------------
# Question 3b: Variance with quadrature
#---------------------------------------------------

function variance_quadrature()
    println("\n=== Question 3b: Variance using Quadrature ===")

    # N(0,2), so true variance is 4
    d = Normal(0, 2)
    σ = 2

    # ∫x²f(x)dx with 7 points
    nodes7, weights7 = lgwt(7, -5*σ, 5*σ)
    variance_7pts = sum(weights7 .* (nodes7.^2) .* pdf.(d, nodes7))

    # ∫x²f(x)dx with 10 points
    nodes10, weights10 = lgwt(10, -5*σ, 5*σ)
    variance_10pts = sum(weights10 .* (nodes10.^2) .* pdf.(d, nodes10))

    println("Variance with 7 quadrature points: ", variance_7pts)
    println("Variance with 10 quadrature points: ", variance_10pts)
    println("True variance: $(σ^2)")

    # the comment is in the script file
end

#---------------------------------------------------
# Question 3c: Monte Carlo practice
#---------------------------------------------------

function practice_monte_carlo()
    println("\n=== Question 3c: Monte Carlo Integration ===")

    σ = 2
    d = Normal(0, σ)
    a, b = -5*σ, 5*σ

    # ∫f(x)dx ≈ (b-a) * mean of f(X_i), with X_i ~ U[a,b]
    function mc_integrate(f, a, b, D)
        draws = rand(D) * (b - a) .+ a  # uniform draws on [a,b]
        return (b - a) * mean(f.(draws))
    end

    for D in [1000, 1000000]
        println("\nWith D = $D draws:")

        # variance: ∫x²f(x)dx
        variance_mc = mc_integrate(x -> x^2 * pdf(d, x), a, b, D)
        println("MC Variance: ", variance_mc, " (true: $(σ^2))")

        # mean: ∫xf(x)dx
        mean_mc = mc_integrate(x -> x * pdf(d, x), a, b, D)
        println("MC Mean: ", mean_mc, " (true: 0)")

        # integral of density: ∫f(x)dx
        density_mc = mc_integrate(x -> pdf(d, x), a, b, D)
        println("MC Density integral: ", density_mc, " (true: 1)")
    end
end

#---------------------------------------------------
# Question 4: Mixed logit with quadrature (DO NOT RUN!)
#---------------------------------------------------

function mixed_logit_quad(theta, X, Z, y, nodes, weights)
    # theta = [21 alphas; mu_gamma; sigma_gamma]
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)

    alpha = theta[1:(K*(J-1))]
    mu_gamma = theta[end-1]     # mean of gamma
    sigma_gamma = theta[end]    # standard deviation of gamma

    # matrix of choice dummies
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = y .== j
    end

    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]

    T = promote_type(eltype(X), eltype(theta))
    P_integrated = zeros(T, N, J)

    # for each node we compute the logit probabilities with gamma_r,
    # and we add them with the weight
    for r in eachindex(nodes)
        gamma_r = mu_gamma + sigma_gamma * nodes[r]

        # logit probabilities, same as Question 1
        num_r = zeros(T, N, J)
        for j = 1:J
            num_r[:,j] = exp.(X * bigAlpha[:,j] .+ gamma_r .* (Z[:,j] .- Z[:,J]))
        end
        dem_r = sum(num_r, dims=2)
        P_r = num_r ./ dem_r

        # weight = quadrature weight * normal density
        density_weight = weights[r] * pdf(Normal(0,1), nodes[r])
        P_integrated .+= P_r * density_weight
    end

    # negative log-likelihood
    loglike = -sum(bigY .* log.(P_integrated))

    return loglike
end


#---------------------------------------------------
# Question 5: Mixed logit with Monte Carlo (DO NOT RUN!)
#---------------------------------------------------

function mixed_logit_mc(theta, X, Z, y, D)
    # same parameters as quadrature
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)

    alpha = theta[1:(K*(J-1))]
    mu_gamma = theta[end-1]
    sigma_gamma = theta[end]

    # matrix of choice dummies
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = y .== j
    end

    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]

    T = promote_type(eltype(X), eltype(theta))
    P_integrated = zeros(T, N, J)

    # abs() because Normal() gives error if sigma_gamma is negative
    gamma_dist = Normal(mu_gamma, abs(sigma_gamma))

    # fixed seed, to have the same draws each time. If not, the optimizer
    # cannot converge
    rng = MersenneTwister(1234)

    # same as quadrature, but with random draws and all weights equal to 1/D
    for d = 1:D
        gamma_d = rand(rng, gamma_dist)

        # logit probabilities, same as Question 1
        num_d = zeros(T, N, J)
        for j = 1:J
            num_d[:,j] = exp.(X * bigAlpha[:,j] .+ gamma_d .* (Z[:,j] .- Z[:,J]))
        end
        dem_d = sum(num_d, dims=2)
        P_d = num_d ./ dem_d

        # average over the draws
        P_integrated .+= P_d / D
    end

    # negative log-likelihood
    loglike = -sum(bigY .* log.(P_integrated))

    return loglike
end


#---------------------------------------------------
# Estimation
#---------------------------------------------------

function optimize_mlogit(X, Z, y)
    K = size(X, 2)
    J = length(unique(y))

    # starting values: estimates of Question 1 of PS3 (21 alphas, then gamma)
    startvals = [ 0.0557,  0.0834, -2.3449,
                  0.0450,  0.7366, -3.1532,
                  0.0926, -0.0842, -4.2733,
                  0.0239,  0.7231, -3.7494,
                  0.0361, -0.6438, -4.2797,
                  0.0853, -1.1714, -6.6787,
                  0.0866, -0.7979, -4.9691,
                 -0.0942]

    # automatic differentiation
    td = TwiceDifferentiable(theta -> mlogit_with_Z(theta, X, Z, y),
                             startvals, autodiff = Optim.ADTypes.AutoForwardDiff())

    result = optimize(td, startvals, LBFGS(),
                     Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true))

    # standard errors from the Hessian
    H  = Optim.hessian!(td, result.minimizer)
    result_se = sqrt.(diag(inv(H)))
    return result.minimizer, result_se
end

# theta_mlogit = estimates of the multinomial logit of Question 1 (21 alphas, then gamma)
function optimize_mixed_logit_quad(X, Z, y, theta_mlogit)
    K = size(X, 2)
    J = length(unique(y))

    # 7 nodes and weights on [-4, 4]
    nodes, weights = lgwt(7, -4, 4)

    # starting values: logit estimates for alphas and mu_gamma, and sigma_gamma = 1
    startvals = [theta_mlogit; 1.0]

    # DO NOT RUN, too slow (the call in allwrap() is commented)
    result = optimize(theta -> mixed_logit_quad(theta, X, Z, y, nodes, weights),
                     startvals, LBFGS(),
                     Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true);
                     autodiff = Optim.ADTypes.AutoForwardDiff())

    return result.minimizer
end

function optimize_mixed_logit_mc(X, Z, y, theta_mlogit)
    K = size(X, 2)
    J = length(unique(y))

    D = 1000  # number of draws

    # starting values: same as quadrature
    startvals = [theta_mlogit; 1.0]

    # DO NOT RUN, too slow (the call in allwrap() is commented)
    result = optimize(theta -> mixed_logit_mc(theta, X, Z, y, D),
                     startvals, LBFGS(),
                     Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true);
                     autodiff = Optim.ADTypes.AutoForwardDiff())

    return result.minimizer
end


#---------------------------------------------------
# Question 6: Main function
#---------------------------------------------------

function allwrap()
    println("=== Problem Set 4: Multinomial and Mixed Logit ===")

    # seed, to have the same Monte Carlo numbers each time
    Random.seed!(1234)

    # load data
    df, X, Z, y = load_data()

    println("Data loaded successfully!")
    println("Sample size: ", size(X, 1))
    println("Number of covariates in X: ", size(X, 2))
    println("Number of alternatives: ", length(unique(y)))

    # Question 1: multinomial logit
    println("\n=== QUESTION 1: MULTINOMIAL LOGIT RESULTS ===")
    theta_hat_mle, theta_hat_se = optimize_mlogit(X, Z, y)
    println("Estimates: ", theta_hat_mle)
    println("Std. Error:", theta_hat_se)

    alpha_hat = theta_hat_mle[1:end-1]
    gamma_hat = theta_hat_mle[end]
    println("γ̂ = ", gamma_hat)

    # Question 2: the answer is in the script file
    println("\n=== QUESTION 2: INTERPRETATION ===")
    println("See the comments in the script file")

    # Question 3: quadrature and Monte Carlo practice
    practice_quadrature()
    variance_quadrature()
    practice_monte_carlo()

    # Question 4: mixed logit with quadrature (NOT RUN, about 4 hours)
    println("\n=== QUESTION 4: MIXED LOGIT QUADRATURE (NOT RUN) ===")
    # theta_hat_quad = optimize_mixed_logit_quad(X, Z, y, theta_hat_mle)
    # println("Estimates: ", theta_hat_quad)

    # Question 5: mixed logit with Monte Carlo (NOT RUN, even longer)
    println("\n=== QUESTION 5: MIXED LOGIT MONTE CARLO (NOT RUN) ===")
    # theta_hat_mc = optimize_mixed_logit_mc(X, Z, y, theta_hat_mle)
    # println("Estimates: ", theta_hat_mc)

    println("\n=== ALL ANALYSES COMPLETE ===")
end
