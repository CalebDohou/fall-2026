#---------------------------------------------------
# ECON 6343: Econometrics III — Problem Set 3 (GEV models)
# Caleb Dohou
#
# Question 5: unit tests for each function of the source file.
#---------------------------------------------------

using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables

cd(@__DIR__)

# Read in the source code
include("PS3_Dohou_source.jl")

# Draw one choice for each row using the matrix of probabilities
function draw_choices(P)
    return [findfirst(cumsum(P[i, :]) .>= rand()) for i in 1:size(P, 1)]
end

@testset "Problem Set 3 Tests" begin

    @testset "load_data" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS3-gev/nlsw88w.csv"
        df, X, Z, y = load_data(url)
        @test size(X) == (2237, 3)
        @test size(Z) == (2237, 8)
        @test length(y) == 2237
        @test sort(unique(y)) == collect(1:8)
    end

    @testset "choice_indicators" begin
        y = [1, 3, 2, 3]
        bigY = choice_indicators(y, 3)
        @test bigY == [1 0 0; 0 0 1; 0 1 0; 0 0 1]
        @test all(sum(bigY, dims=2) .== 1)
    end

    # small fake data to do the tests
    Random.seed!(1)
    N, K, J = 50, 3, 8
    X = randn(N, K)
    Z = randn(N, J)

    @testset "Question 1: mlogit_probs and mlogit_with_Z" begin
        θ = randn(K*(J-1) + 1)
        P = mlogit_probs(θ, X, Z, J)

        @test size(P) == (N, J)
        @test all(P .> 0)
        @test all(isapprox.(sum(P, dims=2), 1.0))

        # if all parameters are 0, all probabilities must be 1/J
        @test all(isapprox.(mlogit_probs(zeros(K*(J-1) + 1), X, Z, J), 1/J))

        # check one observation with the formula of the PS
        i = 1
        β = [reshape(θ[1:end-1], K, J-1) zeros(K)]
        γ = θ[end]
        v = [dot(X[i, :], β[:, j]) + γ*(Z[i, j] - Z[i, J]) for j in 1:J]
        @test isapprox(P[i, :], exp.(v) ./ (1 + sum(exp.(v[1:J-1]))))

        # at 0, negative log-likelihood must be N*log(J)
        y = rand(1:J, N)
        @test isapprox(mlogit_with_Z(zeros(K*(J-1) + 1), X, Z, y), N*log(J))
        @test isapprox(mlogit_with_Z(θ, X, Z, y), -sum(log(P[n, y[n]]) for n in 1:N))
    end

    nesting_structure = [[1, 2, 3], [4, 5, 6, 7]]

    @testset "Question 3: nested_logit_probs and nested_logit_with_Z" begin
        θ = [randn(2K); 0.6; 0.8; 0.3]
        P = nested_logit_probs(θ, X, Z, J, nesting_structure)

        @test size(P) == (N, J)
        @test all(P .> 0)
        @test all(isapprox.(sum(P, dims=2), 1.0))

        # check one observation with the formula of the PS
        i = 2
        βWC, βBC, λWC, λBC, γ = θ[1:K], θ[K+1:2K], θ[2K+1], θ[2K+2], θ[end]
        eWC = [exp((dot(X[i, :], βWC) + γ*(Z[i, j] - Z[i, J])) / λWC) for j in 1:3]
        eBC = [exp((dot(X[i, :], βBC) + γ*(Z[i, j] - Z[i, J])) / λBC) for j in 4:7]
        dem = 1 + sum(eWC)^λWC + sum(eBC)^λBC
        @test isapprox(P[i, 1:3], eWC .* sum(eWC)^(λWC - 1) ./ dem)
        @test isapprox(P[i, 4:7], eBC .* sum(eBC)^(λBC - 1) ./ dem)
        @test isapprox(P[i, 8], 1 / dem)

        # when λ = 1, nested logit become MNL with nest-level betas
        θ1 = [θ[1:2K]; 1.0; 1.0; γ]
        θmnl = [repeat(βWC, 3); repeat(βBC, 4); γ]
        @test isapprox(nested_logit_probs(θ1, X, Z, J, nesting_structure),
                       mlogit_probs(θmnl, X, Z, J))

        y = rand(1:J, N)
        @test isapprox(nested_logit_with_Z(θ, X, Z, y, nesting_structure),
                       -sum(log(P[n, y[n]]) for n in 1:N))
    end

    @testset "mle_se" begin
        # for mean of a N(μ, 1) sample, se must be 1/sqrt(n)
        x = randn(400) .+ 2.0
        obj(θ) = 0.5 * sum((x .- θ[1]).^2)
        μ̂ = [mean(x)]
        @test isapprox(mle_se(obj, μ̂)[1], 1/sqrt(length(x)), rtol=1e-4)
    end

    # with simulated data, estimates should be close to true values
    @testset "Question 1: optimize_mlogit recovers parameters" begin
        Random.seed!(2)
        Ns, Ks = 10_000, 2
        Xs = randn(Ns, Ks)
        Zs = randn(Ns, J)
        θtrue = [0.5*randn(Ks*(J-1)); 0.8]
        ys = draw_choices(mlogit_probs(θtrue, Xs, Zs, J))

        θ̂ = optimize_mlogit(Xs, Zs, ys)
        @test length(θ̂) == Ks*(J-1) + 1
        @test isapprox(θ̂, θtrue, atol=0.2)
        @test isapprox(θ̂[end], θtrue[end], atol=0.1)
    end

    @testset "Question 3: optimize_nested_logit recovers parameters" begin
        Random.seed!(3)
        Ns, Ks = 10_000, 2
        Xs = randn(Ns, Ks)
        Zs = randn(Ns, J)
        θtrue = [0.5, -0.5, 0.3, 0.2, 0.5, 0.7, 0.8]   # β_WC, β_BC, λ_WC, λ_BC, γ
        ys = draw_choices(nested_logit_probs(θtrue, Xs, Zs, J, nesting_structure))

        θ̂ = optimize_nested_logit(Xs, Zs, ys, nesting_structure)
        @test length(θ̂) == 2Ks + 3
        @test isapprox(θ̂, θtrue, atol=0.2)
    end
end
