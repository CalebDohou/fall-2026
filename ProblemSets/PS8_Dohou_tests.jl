#---------------------------------------------------
# ECON 6343: Econometrics III — Problem Set 8 (Factor models)
# Caleb Dohou
#
# Unit tests (Question 7). This file first reads in the source code
# (lgwt.jl and PS8_Dohou_source.jl) and then tests each function.
# The real data is downloaded from the course GitHub URL.
#---------------------------------------------------
using Test, Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta,
      CSV, MultivariateStats, Distributions, FreqTables, ForwardDiff, LineSearches

cd(@__DIR__)

include("lgwt.jl")
include("PS8_Dohou_source.jl")

const URL = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS8-factor/nlsy.csv"

#---------------------------------------------------
# Helper: simulate data from the factor model with known parameters
#---------------------------------------------------
function simulate_factor_data(N::Integer; seed::Integer = 1234)
    rng = MersenneTwister(seed)
    black    = Float64.(rand(rng, N) .< 0.3)
    hispanic = Float64.((rand(rng, N) .< 0.2) .* (1 .- black))
    female   = Float64.(rand(rng, N) .< 0.5)
    schoolt  = rand(rng, N) .* 10
    gradHS   = Float64.(rand(rng, N) .< 0.8)
    grad4yr  = Float64.((rand(rng, N) .< 0.3) .* gradHS)
    ξ = randn(rng, N)

    J = 6
    α = [-0.8 -0.4 -0.7 -0.3 -0.7 -0.9;    # black
         -0.5 -0.4 -0.5 -0.4 -0.5 -0.6;    # hispanic
          0.0  0.4  0.2  0.3  0.3  0.1;    # female
          0.3  0.0  0.2  0.0  0.1  0.3]    # constant
    γ = [0.8, 0.5, 0.8, 0.6, 0.7, 0.7]
    σm = [0.5, 0.8, 0.4, 0.7, 0.6, 0.6]
    β = [-0.2, -0.1, -0.15, 0.01, 0.1, 0.25, 2.0]
    δ = 0.1
    σw = 0.45

    Xfac = hcat(black, hispanic, female, ones(N))
    X    = hcat(black, hispanic, female, schoolt, gradHS, grad4yr, ones(N))
    M    = Xfac * α .+ ξ * γ' .+ randn(rng, N, J) .* σm'
    y    = X * β .+ δ .* ξ .+ σw .* randn(rng, N)

    df = DataFrame(black = black, hispanic = hispanic, female = female, schoolt = schoolt,
                   gradHS = gradHS, grad4yr = grad4yr, logwage = y)
    for (j, c) in enumerate(ASVAB_COLS)
        df[!, c] = M[:, j]
    end
    θtrue = vcat(vec(α), β, γ, δ, σm, σw)
    return df, θtrue
end

# Exact log-likelihood of the factor model: integrating ξ out gives a
# multivariate normal for (asvab_1, ..., asvab_6, logwage) with
# covariance λλ' + diag(σ²), where λ = (γ, δ). Used to check factor_model.
function exact_loglike(θ, X, Xfac, Meas, y)
    K, L, J = size(X, 2), size(Xfac, 2), size(Meas, 2)
    α, β, γ, δ, σ = unpack_params(θ, K, L, J)
    λ = vcat(γ, δ)
    Σ = λ * λ' + Diagonal(σ .^ 2)
    D = MvNormal(zeros(J + 1), Symmetric(Σ))
    resid = hcat(Meas .- Xfac * α, y .- X * β)
    return sum(logpdf(D, resid[i, :]) for i in 1:size(resid, 1))
end

df = load_data(URL)
N  = nrow(df)

@testset "PS8 unit tests" begin

    #-----------------------------------------------
    @testset "lgwt (quadrature)" begin
        x, w = lgwt(25, -5, 5)
        @test length(x) == 25 && length(w) == 25
        @test all(-5 .< x .< 5)
        @test sum(w) ≈ 10 atol = 1e-10                       # integral of 1 over [-5,5]
        @test sum(w .* x .^ 4) ≈ 2 * 5^5 / 5 atol = 1e-8      # exact for polynomials
        @test sum(w .* pdf.(Normal(), x)) ≈ 1 atol = 1e-5      # N(0,1) density integrates to 1
        @test sum(w .* x .^ 2 .* pdf.(Normal(), x)) ≈ 1 atol = 1e-4   # variance of N(0,1)
    end

    #-----------------------------------------------
    @testset "Q1: load_data and base regression" begin
        @test df isa DataFrame
        @test size(df) == (2438, 15)
        for c in [:logwage, :black, :hispanic, :female, :schoolt, :gradHS, :grad4yr, ASVAB_COLS...]
            @test c in propertynames(df)
        end
        @test !any(ismissing, Matrix(df))

        est = estimate_base_regression(df)
        @test length(coef(est)) == 7
        @test coefnames(est) == ["(Intercept)", "black", "hispanic", "female", "schoolt", "gradHS", "grad4yr"]
        # same as OLS by hand
        Xb = hcat(ones(N), Matrix{Float64}(df[:, [:black, :hispanic, :female, :schoolt, :gradHS, :grad4yr]]))
        @test coef(est) ≈ Xb \ df.logwage
    end

    #-----------------------------------------------
    @testset "Q2: ASVAB correlations" begin
        cordf = compute_asvab_correlations(df)
        @test size(cordf) == (6, 7)
        @test cordf.test == String.(ASVAB_COLS)
        C = Matrix{Float64}(cordf[:, ASVAB_COLS])
        @test C ≈ C'                              # symmetric
        @test all(diag(C) .≈ 1)                   # ones on the diagonal
        @test all(-1 .<= C .<= 1)
        @test C ≈ cor(Matrix{Float64}(df[:, ASVAB_COLS]))
        @test cordf[1, :asvabMK] ≈ cor(df.asvabAR, df.asvabMK)
        # extra columns in df do not change the answer
        df_extra = hcat(df, DataFrame(junk = randn(N)))
        @test Matrix{Float64}(compute_asvab_correlations(df_extra)[:, ASVAB_COLS]) ≈ C
    end

    #-----------------------------------------------
    @testset "Q3: full regression with all ASVABs" begin
        est = estimate_full_regression(df)
        @test length(coef(est)) == 13
        @test all(string.(ASVAB_COLS) .∈ Ref(coefnames(est)))
        Xf = hcat(ones(N), Matrix{Float64}(df[:, [:black, :hispanic, :female, :schoolt, :gradHS, :grad4yr, ASVAB_COLS...]]))
        @test coef(est) ≈ Xf \ df.logwage
    end

    #-----------------------------------------------
    @testset "asvab_matrix (J×N, selected by name)" begin
        A = asvab_matrix(df)
        @test size(A) == (6, N)
        @test A[1, :] == df.asvabAR
        @test A[6, :] == df.asvabWK
        # adding a column at the end (like a PC) must not change the matrix
        df_extra = @transform(df, :asvabPCA = randn(N))
        @test asvab_matrix(df_extra) == A
    end

    #-----------------------------------------------
    @testset "Q4: PCA regression" begin
        est, pc1 = estimate_pca_regression(df)
        @test length(pc1) == N
        @test length(coef(est)) == 8
        @test "asvabPCA" in coefnames(est)
        # PC1 has mean zero (PCA centers the data) and is an ability index
        @test abs(mean(pc1)) < 1e-10
        @test abs(cor(pc1, vec(mean(asvab_matrix(df), dims = 1)))) > 0.95
        # PC1 has the largest variance of any unit-length combination of the scores
        A = Matrix{Float64}(df[:, ASVAB_COLS])
        @test var(pc1) ≈ maximum(eigvals(cov(A))) rtol = 1e-6
        # regression coefficients match OLS by hand with pc1
        Xp = hcat(ones(N), Matrix{Float64}(df[:, [:black, :hispanic, :female, :schoolt, :gradHS, :grad4yr]]), pc1)
        @test coef(est) ≈ Xp \ df.logwage
        # df is not modified
        @test !("asvabPCA" in names(df))
    end

    #-----------------------------------------------
    @testset "Q5: factor analysis regression" begin
        est, f1 = estimate_factor_regression(df)
        @test length(f1) == N
        @test length(coef(est)) == 8
        @test "asvabFactor" in coefnames(est)
        @test abs(cor(f1, vec(mean(asvab_matrix(df), dims = 1)))) > 0.95
        Xp = hcat(ones(N), Matrix{Float64}(df[:, [:black, :hispanic, :female, :schoolt, :gradHS, :grad4yr]]), f1)
        @test coef(est) ≈ Xp \ df.logwage
        # same answer when the PC column is already in df (the PS warning)
        _, pc1 = estimate_pca_regression(df)
        df_pc = @transform(df, :asvabPCA = pc1)
        _, f1b = estimate_factor_regression(df_pc)
        @test f1b ≈ f1
        # factor and PC1 are nearly the same variable (up to sign)
        @test abs(cor(f1, pc1)) > 0.95
    end

    #-----------------------------------------------
    @testset "Q6: prepare_factor_matrices" begin
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        @test size(X) == (N, 7)
        @test size(Xfac) == (N, 4)
        @test size(asvabs) == (N, 6)
        @test length(y) == N
        @test all(X[:, end] .== 1) && all(Xfac[:, end] .== 1)   # constant is last
        @test X[:, 1] == df.black && X[:, 4] == df.schoolt && X[:, 6] == df.grad4yr
        @test Xfac[:, 1:3] == Matrix{Float64}(df[:, [:black, :hispanic, :female]])
        @test y == df.logwage
        @test asvabs == Matrix{Float64}(df[:, ASVAB_COLS])
    end

    #-----------------------------------------------
    @testset "Q6: unpack_params" begin
        K, L, J = 7, 4, 6
        θ = collect(1.0:(L*J + K + J + 1 + J + 1))
        @test length(θ) == 45
        α, β, γ, δ, σ = unpack_params(θ, K, L, J)
        @test size(α) == (L, J)
        @test α[:, 1] == [1, 2, 3, 4]          # column-major: θ[1:L] is test 1
        @test α[:, 6] == [21, 22, 23, 24]
        @test β == collect(25.0:31.0)
        @test γ == collect(32.0:37.0)
        @test δ == 38.0 && δ isa Real          # scalar, not a vector
        @test σ == collect(39.0:45.0)
        @test vcat(vec(α), β, γ, δ, σ) == θ    # round trip
    end

    #-----------------------------------------------
    @testset "Q6: starting_values" begin
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        θ0 = starting_values(X, y, Xfac, asvabs)
        @test length(θ0) == 45
        α0, β0, γ0, δ0, σ0 = unpack_params(θ0, 7, 4, 6)
        @test α0[:, 2] ≈ Xfac \ asvabs[:, 2]
        @test β0 ≈ X \ y
        @test all(γ0 .!= 0) && δ0 != 0          # not the all-zero stationary point
        @test all(σ0 .> 0)
    end

    #-----------------------------------------------
    @testset "Q6: factor_model (likelihood)" begin
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        θ0 = starting_values(X, y, Xfac, asvabs)
        nll = factor_model(θ0, X, Xfac, asvabs, y, 25)
        @test nll isa Real
        @test isfinite(nll)

        # matches the exact multivariate normal log-likelihood
        @test -factor_model(θ0, X, Xfac, asvabs, y, 50) ≈ exact_loglike(θ0, X, Xfac, asvabs, y) rtol = 1e-6

        # with zero loadings, the likelihood is a product of independent normals
        K, L, J = 7, 4, 6
        α, β, γ, δ, σ = unpack_params(θ0, K, L, J)
        θz = vcat(vec(α), β, zeros(J), 0.0, σ)
        indep = sum(logpdf.(Normal.(0, σ[j]), asvabs[:, j] .- Xfac * α[:, j]) for j in 1:J) |> sum
        indep += sum(logpdf.(Normal(0, σ[end]), y .- X * β))
        @test -factor_model(θz, X, Xfac, asvabs, y, 25) ≈ indep rtol = 1e-6

        # flipping the sign of all loadings (γ, δ) does not change the likelihood
        θflip = vcat(vec(α), β, -γ, -δ, σ)
        @test factor_model(θflip, X, Xfac, asvabs, y, 25) ≈ nll
        # the sign of σ does not matter either
        θneg = vcat(vec(α), β, γ, δ, -σ)
        @test factor_model(θneg, X, Xfac, asvabs, y, 25) ≈ nll

        # number of quadrature points: R = 25 is close to the exact value
        # (relative error ~6e-6), R = 9 is not (this is why the script uses 25)
        exact = exact_loglike(θ0, X, Xfac, asvabs, y)
        @test -factor_model(θ0, X, Xfac, asvabs, y, 25) ≈ exact rtol = 1e-5
        @test abs(-factor_model(θ0, X, Xfac, asvabs, y, 9) - exact) > 10

        # works with ForwardDiff (needed for Newton), gradient matches finite differences
        f = th -> factor_model(th, X, Xfac, asvabs, y, 25)
        g = ForwardDiff.gradient(f, θ0)
        @test length(g) == 45 && all(isfinite, g)
        h = 1e-6
        for k in (1, 30, 33, 38, 45)
            e = zeros(45); e[k] = h
            @test g[k] ≈ (f(θ0 + e) - f(θ0 - e)) / (2h) rtol = 1e-4
        end
    end

    #-----------------------------------------------
    @testset "Q6: run_estimation recovers true parameters (simulated data)" begin
        dfsim, θtrue = simulate_factor_data(3000)
        X, y, Xfac, asvabs = prepare_factor_matrices(dfsim)
        θ0 = starting_values(X, y, Xfac, asvabs)
        θ̂, se, ll, converged = run_estimation(dfsim, θ0; show_trace = false)
        @test converged
        @test length(θ̂) == 45 && length(se) == 45
        @test all(se .> 0) && all(isfinite, se)
        @test ll ≈ -factor_model(θ̂, X, Xfac, asvabs, y, 25)
        # the MLE beats the starting values and the true parameters
        @test ll >= -factor_model(θ0, X, Xfac, asvabs, y, 25)
        @test ll >= -factor_model(θtrue, X, Xfac, asvabs, y, 25) - 1e-6
        # gradient is zero at the optimum
        @test norm(ForwardDiff.gradient(th -> factor_model(th, X, Xfac, asvabs, y, 25), θ̂)) < 1e-3
        # estimates are close to the truth (σ's up to sign)
        est = copy(θ̂); est[end-6:end] = abs.(est[end-6:end])
        @test maximum(abs.(est .- θtrue) ./ se) < 4.5    # all within ~4 standard errors
        @test maximum(abs.(est .- θtrue)) < 0.15
    end

    #-----------------------------------------------
    @testset "Q6: format_results" begin
        θ = collect(range(-1, 1, length = 45))
        se = fill(0.1, 45)
        asvabs = Matrix{Float64}(df[:, ASVAB_COLS])
        tab = format_results(θ, se, -100.0, asvabs)
        @test tab isa DataFrame
        @test size(tab) == (45, 5)
        @test names(tab) == ["equation", "variable", "coefficient", "std_error", "t_stat"]
        @test tab.equation[1] == "asvabAR" && tab.variable[1] == "black"
        @test tab.equation[24] == "asvabWK" && tab.variable[24] == "constant"
        @test tab.equation[25] == "logwage" && tab.variable[25] == "black"
        @test tab.variable[31] == "constant"
        @test tab.variable[38] == "factor loading (δ)"
        @test tab.equation[45] == "logwage" && tab.variable[45] == "σ"
        @test all(tab.coefficient[39:45] .>= 0)            # σ's reported as |σ|
        @test tab.coefficient[1:38] == θ[1:38]
        @test tab.t_stat ≈ tab.coefficient ./ tab.std_error
    end
end
