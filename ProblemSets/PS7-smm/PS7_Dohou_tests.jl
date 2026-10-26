#---------------------------------------------------
# ECON 6343: Econometrics III — Problem Set 7 (GMM and SMM)
# Caleb Dohou
#
# Unit tests for the functions in PS7_Dohou_source.jl (Question 7).
# Reads in the source code first, then runs the tests.
# Tests use small samples so they run quickly; main() is not run here
# (it is run by PS7_Dohou_script.jl).
#---------------------------------------------------
using Test, Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM

cd(@__DIR__)

include("PS7_Dohou_source.jl")

const URL = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS1-julia-intro/nlsw88.csv"

# Helper: choice-indicator matrix d (N×J)
choice_matrix(y, J) = hcat([y .== j for j in 1:J]...) .* 1.0

# Helper: logit probabilities P (N×J) for K*(J-1) parameter vector α
function logit_probs(α, X)
    K = size(X, 2)
    J = length(α) ÷ K + 1
    V = X * [reshape(α, K, J-1) zeros(K)]
    return exp.(V) ./ sum(exp.(V); dims=2)
end

@testset "PS7 unit tests" begin

    #-----------------------------------------------
    # Data functions
    #-----------------------------------------------
    @testset "load_data" begin
        df, X, y = load_data(URL)
        @test size(X, 1) == nrow(df) == length(y)
        @test size(X, 2) == 4
        @test all(X[:, 1] .== 1)                      # intercept
        @test X[:, 2] == df.age
        @test all(in.(X[:, 3], Ref((0.0, 1.0))))      # race == 1 dummy
        @test all(in.(X[:, 4], Ref((0.0, 1.0))))      # collgrad == 1 dummy
        @test X[:, 3] == (df.race .== 1)
        @test y ≈ log.(df.wage)                       # y is log wage
        @test !any(ismissing, y)
    end

    @testset "prepare_occupation_data" begin
        df0, _, _ = load_data(URL)
        df, X, y = prepare_occupation_data(df0)
        @test !any(ismissing, df.occupation)
        @test sort(unique(y)) == collect(1:7)         # 8-13 collapsed into 7
        @test size(X) == (length(y), 4)
        @test all(X[:, 1] .== 1)
        @test X[:, 3] == (df.race .== 1)              # white dummy
        @test eltype(df.white) == Bool
        @test nrow(df) == nrow(df0) - sum(ismissing.(df0.occupation))
    end

    #-----------------------------------------------
    # Question 1: OLS via GMM
    #-----------------------------------------------
    @testset "ols_gmm" begin
        Random.seed!(1)
        N, K = 500, 3
        X = [ones(N) randn(N, K-1)]
        β = [1.0, -0.5, 2.0]
        y = X * β + 0.1 * randn(N)

        # Objective equals the sum of squared residuals
        b = randn(K)
        @test ols_gmm(b, X, y) ≈ sum((y - X*b).^2)
        @test ols_gmm(b, X, y) >= 0

        # Zero when the data fit exactly
        @test ols_gmm(β, X, X*β) ≈ 0 atol=1e-20

        # Minimizer equals closed-form OLS
        res = optimize(b -> ols_gmm(b, X, y), zeros(K), LBFGS(), Optim.Options(g_tol=1e-8))
        @test res.minimizer ≈ inv(X'X) * X'y atol=1e-6
    end

    #-----------------------------------------------
    # Question 2: MLE and GMM for multinomial logit
    #-----------------------------------------------
    Random.seed!(2)
    # Large N because alternative 2 is rare (~1.6% of choices) and its
    # coefficients are noisy in smaller samples
    ySim, XSim = sim_logit_w_gumbel(100_000, 4)  # true β = true_beta(4), K = 3, J = 4
    α_true = vec(true_beta(4)[:, 1:3])
    N, K, J = length(ySim), 3, 4

    @testset "mlogit_mle" begin
        # At α = 0 all choices have probability 1/J, so -LL = N*log(J)
        @test mlogit_mle(zeros(K*(J-1)), XSim, ySim) ≈ N * log(J)

        # Matches the log-likelihood computed by hand
        α = 0.1 * randn(K*(J-1))
        P = logit_probs(α, XSim)
        d = choice_matrix(ySim, J)
        @test mlogit_mle(α, XSim, ySim) ≈ -sum(d .* log.(P))
        @test mlogit_mle(α, XSim, ySim) > 0

        # The true parameters fit better than α = 0
        @test mlogit_mle(α_true, XSim, ySim) < mlogit_mle(zeros(K*(J-1)), XSim, ySim)

        # Recovers the true parameters
        res = optimize(a -> mlogit_mle(a, XSim, ySim), zeros(K*(J-1)), LBFGS(),
                       Optim.Options(g_tol=1e-5))
        @test Optim.converged(res)
        @test maximum(abs.(res.minimizer .- α_true)) < 0.15
    end

    # MLE on the simulated data, reused below
    α_mle = optimize(a -> mlogit_mle(a, XSim, ySim), zeros(K*(J-1)), LBFGS(),
                     Optim.Options(g_tol=1e-6)).minimizer

    @testset "mlogit_gmm (just-identified)" begin
        # Its moments are the MLE first-order conditions: objective ≈ 0 at the MLE
        @test mlogit_gmm(α_mle, XSim, ySim) < 1e-6
        @test mlogit_gmm(zeros(K*(J-1)), XSim, ySim) > 1

        # Matches N * g'g with g_jk = mean((d_ij - P_ij) * X_ik)
        α = 0.1 * randn(K*(J-1))
        P = logit_probs(α, XSim)
        d = choice_matrix(ySim, J)
        g = vec(XSim' * (d[:, 1:J-1] .- P[:, 1:J-1]) ./ N)
        @test mlogit_gmm(α, XSim, ySim) ≈ N * dot(g, g)
    end

    @testset "mlogit_gmm_overid" begin
        # At α = 0: each person adds (1-1/J)^2 + (J-1)/J^2 = (J-1)/J
        @test mlogit_gmm_overid(zeros(K*(J-1)), XSim, ySim) ≈ N * (J-1) / J

        # Matches g'g with g = vec(d - P)
        α = 0.1 * randn(K*(J-1))
        g = vec(choice_matrix(ySim, J) .- logit_probs(α, XSim))
        @test mlogit_gmm_overid(α, XSim, ySim) ≈ dot(g, g)
        @test mlogit_gmm_overid(α, XSim, ySim) >= 0

        # Minimizer from the MLE start is close to the MLE but has a lower objective
        res = optimize(a -> mlogit_gmm_overid(a, XSim, ySim), α_mle, LBFGS(),
                       Optim.Options(g_tol=1e-5))
        @test res.minimum <= mlogit_gmm_overid(α_mle, XSim, ySim)
        @test maximum(abs.(res.minimizer .- α_true)) < 0.3
    end

    #-----------------------------------------------
    # Question 3: Simulation
    #-----------------------------------------------
    @testset "true_beta" begin
        β4 = true_beta(4)
        @test size(β4) == (3, 4)
        @test β4[:, 4] == zeros(3)                    # normalization
        @test β4[:, 1] == [1, -1, 0.5]

        β3 = true_beta(3)
        @test size(β3) == (3, 3)
        @test β3[:, 3] == zeros(3)
        @test all(-1 .<= β3[:, 1:2] .<= 1)
    end

    @testset "sim_logit" begin
        Random.seed!(3)
        Y, X = sim_logit(50_000, 4)
        @test length(Y) == 50_000
        @test size(X) == (50_000, 3)
        @test all(X[:, 1] .== 1)
        @test eltype(Y) <: Integer
        @test all(1 .<= Y .<= 4)

        # Choice shares match the average model probabilities
        P̄ = vec(mean(logit_probs(α_true, X); dims=1))
        shares = [mean(Y .== j) for j in 1:4]
        @test maximum(abs.(shares .- P̄)) < 0.01

        # Works for other J
        Y3, X3 = sim_logit(1_000, 3)
        @test all(1 .<= Y3 .<= 3)
        @test size(X3, 2) == 3
    end

    @testset "sim_logit_w_gumbel" begin
        Random.seed!(4)
        Y, X = sim_logit_w_gumbel(50_000, 4)
        @test length(Y) == 50_000
        @test size(X) == (50_000, 3)
        @test all(1 .<= Y .<= 4)

        P̄ = vec(mean(logit_probs(α_true, X); dims=1))
        shares = [mean(Y .== j) for j in 1:4]
        @test maximum(abs.(shares .- P̄)) < 0.01

        # Both simulators give the same choice shares
        Y1, _ = sim_logit(50_000, 4)
        @test maximum(abs.([mean(Y .== j) - mean(Y1 .== j) for j in 1:4])) < 0.01

        Y5, _ = sim_logit_w_gumbel(1_000, 5)
        @test all(1 .<= Y5 .<= 5)
    end

    #-----------------------------------------------
    # Question 5: SMM
    #-----------------------------------------------
    @testset "mlogit_smm_overid" begin
        n = 2_000
        Xs, ys = XSim[1:n, :], ySim[1:n]
        α = α_true

        # Deterministic: same seed gives the same value
        @test mlogit_smm_overid(α, Xs, ys, 50) == mlogit_smm_overid(α, Xs, ys, 50)
        # A different seed gives different draws
        @test mlogit_smm_overid(α, Xs, ys, 50) != mlogit_smm_overid(α, Xs, ys, 50; seed=99)
        # Does not reset the global random number generator
        Random.seed!(5); r1 = rand()
        Random.seed!(5); mlogit_smm_overid(α, Xs, ys, 10); r2 = rand()
        @test r1 == r2
        @test mlogit_smm_overid(α, Xs, ys, 50) >= 0

        # With many draws it approaches the analytic GMM objective
        @test isapprox(mlogit_smm_overid(α, Xs, ys, 2_000),
                       mlogit_gmm_overid(α, Xs, ys); rtol=0.01)

        # At α = 0 the simulated frequencies are ≈ 1/J for everyone
        @test isapprox(mlogit_smm_overid(zeros(K*(J-1)), Xs, ys, 2_000),
                       n * (J-1) / J; rtol=0.01)

        # The true parameters fit better than α = 0
        @test mlogit_smm_overid(α, Xs, ys, 200) < mlogit_smm_overid(zeros(K*(J-1)), Xs, ys, 200)
    end
end;
