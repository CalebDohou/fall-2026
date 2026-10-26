################################################################################
# ECON 6343: Econometrics III — Problem Set 7 (GMM and SMM)
# Caleb Dohou
#
# Source file: all the functions. We run everything from PS7_Dohou_script.jl.
################################################################################

################################################################################
# Data functions
################################################################################

"""
    load_data(url)

Load the nlsw88 data and make X and y for the OLS regression.
y is log wage. Return df, X, y.
"""
function load_data(url)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    df = @transform(df, :log_wage = log.(:wage))
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.log_wage
    return df, X, y
end


"""
    prepare_occupation_data(df)

Make the data for the multinomial logit. Occupations 8 to 13 are put
in occupation 7. Return df, X, y.
"""
function prepare_occupation_data(df)
    df = dropmissing(df, :occupation)
    df[df.occupation.==8 ,:occupation] .= 7
    df[df.occupation.==9 ,:occupation] .= 7
    df[df.occupation.==10,:occupation] .= 7
    df[df.occupation.==11,:occupation] .= 7
    df[df.occupation.==12,:occupation] .= 7
    df[df.occupation.==13,:occupation] .= 7
    df.white = df.race .== 1
    X = [ones(size(df,1),1) df.age df.white df.collgrad]
    y = df.occupation
    return df, X, y
end

################################################################################
# Question 1: OLS by GMM
################################################################################

"""
    ols_gmm(β, X, y)

GMM objective for OLS. The moment is g = y - Xβ and W = I,
so the objective is g'g.
"""
function ols_gmm(β, X, y)
    # predicted values
    ŷ = X*β

    # moments
    g = y - ŷ

    # objective g'Ig = g'g (g⋅g gives a scalar)
    J = g⋅g

    return J
end

################################################################################
# Question 2: Multinomial logit by MLE and GMM
################################################################################

"""
    mlogit_mle(alpha, X, y)

Negative log likelihood of the multinomial logit.
alpha has K*(J-1) elements, the coefficients of the last choice are 0.
"""
function mlogit_mle(alpha, X, y)
    N = size(X,1)              # number of obs
    K = size(X,2)              # number of covariates
    J = length(alpha) ÷ K + 1  # number of choices

    # make the K×J matrix of coefficients, last column is 0
    alpha_mat = [reshape(alpha, K, J-1) zeros(K, 1)]

    # d matrix (1 if person i chooses j)
    d = zeros(N, J)
    for j = 1:J
        d[:,j] .= y .== j
    end

    # probabilities
    p = zeros(N, J)
    for j = 1:J
        p[:,j] .= exp.(X * alpha_mat[:, j])./sum(exp.(X * alpha_mat); dims=2)
    end

    # negative log likelihood
    loglike = -sum(d .* log.(p))

    return loglike
end

"""
    mlogit_gmm(α, X, y)

Just-identified GMM for the multinomial logit.
Moments: mean of X_ik*(d_ij - P_ij), so K*(J-1) moments for K*(J-1)
parameters. These are the MLE first order conditions, so we get the MLE.
"""
function mlogit_gmm(α, X, y)
    K = size(X, 2)
    J = length(α) ÷ K + 1 # α has K*(J-1) elements
    N = length(y)

    # choice indicators
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:,j] = y .== j
    end

    # coefficients matrix
    bigα = [reshape(α, K, J-1) zeros(K, 1)]

    # probabilities
    P = exp.(X * bigα) ./ sum(exp.(X * bigα); dims=2)

    # moments g, dimension K*(J-1)
    g = zeros((J-1)*K)
    for j = 1:(J-1)
        for k = 1:K
            g[(j-1)*K + k] = mean((bigY[:,j] .- P[:,j]) .* X[:,k])
        end
    end

    # objective N*g'g (we don't call it J because J is the number of choices)
    obj = N * (g⋅g)

    return obj
end

"""
    mlogit_gmm_overid(α, X, y)

Over-identified GMM for the multinomial logit (Question 2b and 2c).
g = vec(d - P) has N*J elements, and W = I.
"""
function mlogit_gmm_overid(α, X, y)
    K = size(X, 2)
    J = length(α) ÷ K + 1 # α has K*(J-1) elements
    N = length(y)

    # choice indicators
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:,j] = y .== j
    end

    # coefficients matrix and probabilities
    bigα = [reshape(α, K, J-1) zeros(K, 1)]
    P = zeros(N, J)
    for j = 1:J
        P[:,j] .= exp.(X * bigα[:, j])./sum(exp.(X * bigα); dims=2)
    end

    # stack the moments in a vector of size N*J
    g = bigY[:] .- P[:]

    # objective g'Wg with W = I
    obj = g⋅g

    return obj
end

################################################################################
# Question 3: Simulate multinomial logit data
################################################################################

"""
    true_beta(J)

The true β (3×J) for the simulations. The last column is 0.
For J = 4 the values are fixed, for other J they are random in [-1, 1].
"""
function true_beta(J)
    if J == 4
        return hcat([1, -1, 0.5], [-2, 0.5, 0.3], [0, -0.5, 2], zeros(3))
    else
        return hcat(2 .* rand(3, J-1) .- 1, zeros(3))
    end
end

"""
    sim_logit(N, J)

Simulate logit data with a uniform draw:
Y_i = sum over j of 1[ sum_{k=j}^J P_ik > ε_i ], with ε_i ~ U[0,1].
X has an intercept and 2 normal covariates (K = 3). Return Y, X.
"""
function sim_logit(N=100_000, J=4)
    # X: intercept, N(0,1) and N(2,4)
    X = hcat(ones(N), randn(N), 2 .+ 2 .* randn(N))

    # true coefficients
    β = true_beta(J)

    # probabilities
    P = zeros(N, J)
    for j = 1:J
        P[:,j] .= exp.(X * β[:, j])./sum(exp.(X * β); dims=2)
    end

    # uniform shocks
    draw = rand(N)

    # choices
    Y = zeros(Int, N)
    for j = 1:J
        Y += vec(draw .< sum(P[:,j:J]; dims=2))
    end

    return Y, X
end

"""
    sim_logit_w_gumbel(N, J)

Simulate logit data with T1EV (Gumbel) shocks:
Y_i = argmax_j X_iβ_j + ε_ij. Return Y, X.
"""
function sim_logit_w_gumbel(N=100_000, J=4)
    # X: intercept, N(0,1) and N(2,4)
    X = hcat(ones(N), randn(N), 2 .+ 2 .* randn(N))

    # true coefficients
    β = true_beta(J)

    # Gumbel shocks
    ε = rand(Gumbel(0,1), N, J)

    # choose the alternative with the highest utility
    Y = argmax.(eachrow(X * β .+ ε))

    return Y, X
end

################################################################################
# Question 5: Multinomial logit by SMM
################################################################################

"""
    mlogit_smm_overid(α, X, y, D; seed=1234)

SMM objective for the multinomial logit (slide 18 of Lecture 9).
We simulate the choices D times with Gumbel shocks (like in Question 3),
take the average to get simulated frequencies, and use g = vec(d - P̃)
with W = I. We use the same draws for every α (fixed seed), so the
objective does not change randomly between evaluations.
"""
function mlogit_smm_overid(α, X, y, D; seed=1234)
    K = size(X, 2)
    J = length(α) ÷ K + 1 # α has K*(J-1) elements
    N = length(y)

    # actual choice indicators
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:,j] = y .== j
    end

    # simulated choice frequencies
    bigỸ = zeros(N, J)

    # coefficients matrix and utilities
    bigα = [reshape(α, K, J-1) zeros(K, 1)]
    Xα = X * bigα

    # simulate D times (with loops, it is faster than argmax.(eachrow(...)))
    rng = MersenneTwister(seed)
    gumbel = Gumbel(0,1)
    for d = 1:D
        for i = 1:N
            # draw the shocks and find the choice with the highest utility
            best_j = 1
            best_u = Xα[i,1] + rand(rng, gumbel)
            for j = 2:J
                u = Xα[i,j] + rand(rng, gumbel)
                if u > best_u
                    best_u = u
                    best_j = j
                end
            end
            # add to the frequencies
            bigỸ[i, best_j] += 1/D
        end
    end

    # moments (actual - simulated)
    g = bigY[:] .- bigỸ[:]

    # objective g'g (W = I like in mlogit_gmm_overid)
    obj = g⋅g

    return obj
end

################################################################################
# Question 6: main function
################################################################################

"""
    main(; D_smm=1000, smm_iters=1000)

Run all the questions. D_smm is the number of draws for SMM and
smm_iters is the max number of Nelder-Mead iterations.
Return all the estimates.
"""
function main(; D_smm=1000, smm_iters=1000)
    println("="^80)
    println("Problem Set 7: GMM and SMM Estimation")
    println("="^80)

    # load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df, X_wage, y_wage = load_data(url)
    df, X, y = prepare_occupation_data(df)

    #--------------------------------------------------------------------------
    # Question 1: OLS by GMM
    #--------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 1: OLS Estimation via GMM")
    println("="^80)

    β_hat_gmm = optimize(b -> ols_gmm(b, X_wage, y_wage),
                         rand(size(X_wage,2)),
                         LBFGS(),
                         Optim.Options(g_tol=1e-6, iterations=100_000))

    # closed form OLS: (X'X)^(-1)X'y
    β_ols = inv(X_wage'X_wage) * X_wage'y_wage

    println("GMM estimates: ", round.(β_hat_gmm.minimizer, digits=4))
    println("OLS estimates: ", round.(β_ols, digits=4))
    println("Max abs difference: ", maximum(abs.(β_hat_gmm.minimizer .- β_ols)))

    #--------------------------------------------------------------------------
    # Question 2: Multinomial logit by MLE and GMM
    #--------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 2: Multinomial Logit via MLE and GMM")
    println("="^80)

    # starting values from LPMs (one for each occupation)
    for j = 1:7
        tempname = Symbol(string("occ", j))
        df[:, tempname] = df.occupation .== j
    end
    svals = zeros(size(X,2), 7)
    for j = 1:7
        tempname = Symbol(string("occ", j))
        f = term(tempname) ~ term(:age) + term(:white) + term(:collgrad)
        svals[:,j] = coef(lm(f, df))
    end
    # difference with the base category (7)
    svals = svals[:,1:6] .- svals[:,7]
    svals = svals[:]

    # 2(a) MLE
    α_hat_mle = optimize(a -> mlogit_mle(a, X, y),
                         svals,
                         LBFGS(),
                         Optim.Options(g_tol=1e-5, iterations=100_000))
    println("2(a) MLE estimates:\n", round.(reshape(α_hat_mle.minimizer, 4, 6), digits=4))

    # 2(b) GMM, MLE estimates as starting values
    α_hat_gmm_mle_start = optimize(a -> mlogit_gmm_overid(a, X, y),
                                   α_hat_mle.minimizer,
                                   LBFGS(),
                                   Optim.Options(g_tol=1e-5, iterations=100_000))
    println("2(b) GMM estimates (MLE start):\n",
            round.(reshape(α_hat_gmm_mle_start.minimizer, 4, 6), digits=4))

    # 2(c) GMM, random starting values
    α_hat_gmm_random_start = optimize(a -> mlogit_gmm_overid(a, X, y),
                                      rand(length(svals)),
                                      LBFGS(),
                                      Optim.Options(g_tol=1e-5, iterations=100_000))
    println("2(c) GMM estimates (random start):\n",
            round.(reshape(α_hat_gmm_random_start.minimizer, 4, 6), digits=4))

    # compare (b) and (c)
    println("GMM objective (MLE start):    ", α_hat_gmm_mle_start.minimum)
    println("GMM objective (random start): ", α_hat_gmm_random_start.minimum)
    println("Max abs difference in estimates (b vs c): ",
            maximum(abs.(α_hat_gmm_mle_start.minimizer .- α_hat_gmm_random_start.minimizer)))

    #--------------------------------------------------------------------------
    # Question 3: Simulate data and recover the parameters
    #--------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 3: Simulate Data and Recover Parameters")
    println("="^80)

    ySim, XSim = sim_logit(100_000, 4)
    ySimG, XSimG = sim_logit_w_gumbel(100_000, 4)

    println("Note: X and ε drawn separately so they won't match exactly,")
    println("but with the same β, distribution of X and large N they should be close")
    println("freq of each choice in ySim:  ", round.([mean(ySim.==j) for j in 1:4], digits=4))
    println("freq of each choice in ySimG: ", round.([mean(ySimG.==j) for j in 1:4], digits=4))

    # estimate with MLE (3 covariates × 3 choices that are not the base)
    α_hat_sim = optimize(a -> mlogit_mle(a, XSim, ySim),
                         rand(9),
                         LBFGS(),
                         Optim.Options(g_tol=1e-5, iterations=100_000))
    α_hat_simG = optimize(a -> mlogit_mle(a, XSimG, ySimG),
                          rand(9),
                          LBFGS(),
                          Optim.Options(g_tol=1e-5, iterations=100_000))

    β_true = true_beta(4)
    println("True β (first J-1 columns):\n", β_true[:, 1:3])
    println("Estimated β, method 1 (sim_logit):\n",
            round.(reshape(α_hat_sim.minimizer, 3, 3), digits=3))
    println("Estimated β, method 2 (sim_logit_w_gumbel):\n",
            round.(reshape(α_hat_simG.minimizer, 3, 3), digits=3))

    #--------------------------------------------------------------------------
    # Question 5: Multinomial logit by SMM
    #--------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 5: Multinomial Logit via SMM (D = $D_smm)")
    println("="^80)

    # the SMM objective is not smooth (it comes from argmax),
    # so we use Nelder-Mead and not LBFGS
    α_hat_smm = optimize(th -> mlogit_smm_overid(th, X, y, D_smm),
                         α_hat_mle.minimizer,  # MLE as starting values
                         NelderMead(),
                         Optim.Options(g_tol=1e-6, iterations=smm_iters))

    println("SMM converged: ", Optim.converged(α_hat_smm),
            " (iterations used: ", Optim.iterations(α_hat_smm), " of ", smm_iters,
            ", objective evaluations: ", Optim.f_calls(α_hat_smm), ")")
    println("SMM objective at MLE start: ", mlogit_smm_overid(α_hat_mle.minimizer, X, y, D_smm))
    println("SMM objective at SMM estimates: ", α_hat_smm.minimum)
    println("MLE estimates:\n", round.(reshape(α_hat_mle.minimizer, 4, 6), digits=4))
    println("GMM estimates:\n", round.(reshape(α_hat_gmm_mle_start.minimizer, 4, 6), digits=4))
    println("SMM estimates:\n", round.(reshape(α_hat_smm.minimizer, 4, 6), digits=4))
    println("Max abs difference SMM vs MLE: ",
            maximum(abs.(α_hat_smm.minimizer .- α_hat_mle.minimizer)))

    println("\n" * "="^80)
    println("Estimation Complete!")
    println("="^80)

    # return the estimates
    return (β_gmm = β_hat_gmm.minimizer, β_ols = β_ols,
            α_mle = α_hat_mle.minimizer,
            α_gmm_mle_start = α_hat_gmm_mle_start.minimizer,
            α_gmm_random_start = α_hat_gmm_random_start.minimizer,
            α_sim = α_hat_sim.minimizer, α_simG = α_hat_simG.minimizer,
            α_smm = α_hat_smm.minimizer)
end
