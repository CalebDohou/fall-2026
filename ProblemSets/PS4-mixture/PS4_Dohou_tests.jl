#---------------------------------------------------
# ECON 6343: Econometrics III — Problem Set 4 (Mixed Logit Models)
# Caleb Dohou
#
# Question 7: unit tests for each function of the source file.
#---------------------------------------------------

using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, Distributions, ForwardDiff

cd(@__DIR__)

# Read in the source code
include("lgwt.jl")
include("PS4_Dohou_source.jl")

#---------------------------------------------------
# Helper functions for the tests
#---------------------------------------------------

# Logit probabilities (N x J) computed with the formula of the PS, to compare
function logit_probs(θ, X, Z)
    K, J = size(X, 2), size(Z, 2)
    β = [reshape(θ[1:end-1], K, J-1) zeros(K)]
    v = X * β .+ θ[end] .* (Z .- Z[:, J])
    return exp.(v) ./ sum(exp.(v), dims=2)
end

# Draw one choice for each row using the matrix of probabilities
function draw_choices(P)
    return [findfirst(cumsum(P[i, :]) .>= rand()) for i in 1:size(P, 1)]
end

# Run a function without showing what it prints (the optimizers print the trace)
function quiet(f)
    return redirect_stdout(f, devnull)
end

# Run a function and return what it prints as a String
function capture(f)
    path, io = mktemp()
    try
        redirect_stdout(f, io)
    finally
        close(io)
    end
    out = read(path, String)
    rm(path)
    return out
end

# Find the numbers that are printed after a label, for example "MC Mean: 0.04"
function printed_numbers(out, label)
    return [parse(Float64, m.captures[1]) for m in eachmatch(Regex(label * raw"\s*(-?[\d.]+(?:e-?\d+)?)"), out)]
end

# Gradient with finite differences, to compare with automatic differentiation
function fd_gradient(f, θ; h=1e-6)
    return [(f(θ .+ h .* (1:length(θ) .== k)) - f(θ .- h .* (1:length(θ) .== k))) / (2h) for k in 1:length(θ)]
end

@testset "Problem Set 4 Tests" begin

    #---------------------------------------------------
    # Source file
    #---------------------------------------------------
    @testset "source file" begin
        # all the functions exist
        for f in (load_data, mlogit_with_Z, practice_quadrature, variance_quadrature,
                  practice_monte_carlo, mixed_logit_quad, mixed_logit_mc, optimize_mlogit,
                  optimize_mixed_logit_quad, optimize_mixed_logit_mc, allwrap)
            @test f isa Function
        end

        # the mixed logit estimations must NOT run (4 hours): every line that
        # calls the two optimizers must be a comment
        lines = readlines("PS4_Dohou_source.jl")
        calls = filter(l -> occursin(r"=\s*optimize_mixed_logit_(quad|mc)\(", l), lines)
        @test length(calls) == 2
        @test all(startswith.(strip.(calls), "#"))

        # no TODO of the starter is left
        @test !any(l -> occursin("TODO", l), lines)
        @test !any(l -> occursin("FILL IN", l), lines)
    end

    #---------------------------------------------------
    # lgwt
    #---------------------------------------------------
    @testset "lgwt" begin
        nodes, weights = lgwt(7, -4, 4)
        @test length(nodes) == 7
        @test length(weights) == 7
        @test all(-4 .< nodes .< 4)
        @test all(weights .> 0)
        @test length(unique(nodes)) == 7
        # weights sum to the length of the interval
        @test isapprox(sum(weights), 8.0)
        # on a symmetric interval nodes and weights are symmetric around 0
        p = sortperm(nodes)
        @test isapprox(nodes[p], -reverse(nodes[p]), atol=1e-12)
        @test isapprox(weights[p], reverse(weights[p]))
        # default interval is [-1, 1]
        n1, w1 = lgwt(5)
        @test isapprox(sum(w1), 2.0)
        @test all(-1 .< n1 .< 1)

        # quadrature with n points is exact for polynomials until degree 2n-1
        n4, w4 = lgwt(4, -1, 2)
        @test isapprox(sum(w4 .* n4.^2), 3.0)          # ∫x²dx on [-1,2] = 3
        @test isapprox(sum(w4 .* n4.^6), 129/7)        # degree 6 < 2*4-1
        @test isapprox(sum(w4 .* n4.^7), 255/8)        # degree 7 = 2*4-1
        # ... and not exact for degree 2n: 2 points, ∫x⁴dx on [-1,1] = 2/5
        n2, w2 = lgwt(2, -1, 1)
        @test !isapprox(sum(w2 .* n2.^4), 2/5)
        @test isapprox(sum(w2 .* n2.^4), 2/9)

        # more points give better approximation of a function that is not polynomial
        err(n) = abs(sum(lgwt(n, -4, 4)[2] .* pdf.(Normal(0, 1), lgwt(n, -4, 4)[1])) - (cdf(Normal(0, 1), 4) - cdf(Normal(0, 1), -4)))
        @test err(20) < err(10) < err(5)
        @test err(20) < 1e-8
    end

    # small fake data to do the tests
    # (y must contain all the J alternatives, because the functions use length(unique(y)))
    Random.seed!(1)
    N, K, J = 200, 3, 8
    X = randn(N, K)
    Z = randn(N, J)
    y = [collect(1:J); rand(1:J, N - J)]
    θ = 0.5 .* randn(K*(J-1) + 1)
    bigY = [y[n] == j for n in 1:N, j in 1:J]

    #---------------------------------------------------
    # Question 1
    #---------------------------------------------------
    @testset "Question 1: mlogit_with_Z (values)" begin
        ll = mlogit_with_Z(θ, X, Z, y)
        @test ll isa Float64
        @test isfinite(ll)
        @test ll > 0

        # at 0, all probabilities are 1/J so negative log-likelihood must be N*log(J)
        @test isapprox(mlogit_with_Z(zeros(K*(J-1) + 1), X, Z, y), N*log(J))

        # compare with the formula of the PS
        P = logit_probs(θ, X, Z)
        @test isapprox(ll, -sum(log(P[n, y[n]]) for n in 1:N))

        # check one observation alone with the formula of the PS
        i = 5
        β = [reshape(θ[1:end-1], K, J-1) zeros(K)]
        v = [dot(X[i, :], β[:, j]) + θ[end]*(Z[i, j] - Z[i, J]) for j in 1:J]
        p_i = exp.(v) ./ (1 + sum(exp.(v[1:J-1])))
        @test isapprox(sum(p_i), 1.0)
        # likelihood of all the data without observation i, plus observation i
        keep = setdiff(1:N, i)
        @test isapprox(ll, mlogit_with_Z(θ, X[keep, :], Z[keep, :], y[keep]) - log(p_i[y[i]]))

        # likelihood is a sum: two halves of the data add up
        # (first half has all J alternatives; for second half we check it has too)
        h1, h2 = 1:100, 101:N
        @test length(unique(y[h2])) == J
        @test isapprox(ll, mlogit_with_Z(θ, X[h1, :], Z[h1, :], y[h1]) + mlogit_with_Z(θ, X[h2, :], Z[h2, :], y[h2]))

        # the order of the observations does not matter
        perm = randperm(N)
        @test isapprox(ll, mlogit_with_Z(θ, X[perm, :], Z[perm, :], y[perm]))
    end

    @testset "Question 1: mlogit_with_Z (properties of the model)" begin
        ll = mlogit_with_Z(θ, X, Z, y)

        # γ must matter (in the draft it was not in the likelihood)
        θ2 = copy(θ); θ2[end] += 1.0
        @test mlogit_with_Z(θ2, X, Z, y) != ll

        # only differences of Z matter: we can add any constant to all wages of a person
        c = randn(N)
        @test isapprox(mlogit_with_Z(θ, X, Z .+ c, y), ll)

        # if all the Z are the same for a person, γ does not matter
        Zsame = repeat(randn(N), 1, J)
        @test isapprox(mlogit_with_Z(θ, X, Zsame, y), mlogit_with_Z(θ2, X, Zsame, y))

        # works with integer X like in the real data (age, white, collgrad)
        Xint = hcat(rand(18:45, N), rand(0:1, N), rand(0:1, N))
        θsmall = 0.05 .* θ
        @test isapprox(mlogit_with_Z(θsmall, Xint, Z, y), mlogit_with_Z(θsmall, Float64.(Xint), Z, y))

        # when one alternative is much better, people who choose it have likelihood ≈ 0
        # (big γ and everybody choose the alternative with highest Z)
        ybest = [argmax(Z[n, :]) for n in 1:N]
        ybest[1:J] = 1:J   # keep all the alternatives in y
        θbig = [zeros(K*(J-1)); 50.0]
        @test mlogit_with_Z(θbig, X[J+1:end, :], Z[J+1:end, :], ybest[J+1:end]) <
              mlogit_with_Z(zeros(K*(J-1) + 1), X[J+1:end, :], Z[J+1:end, :], ybest[J+1:end])
    end

    @testset "Question 1: mlogit_with_Z (derivatives)" begin
        f = t -> mlogit_with_Z(t, X, Z, y)

        # automatic differentiation works and agree with finite differences
        g = ForwardDiff.gradient(f, θ)
        @test length(g) == K*(J-1) + 1
        @test isapprox(g, fd_gradient(f, θ), atol=1e-4)

        # ... and with the formula of the score of the logit:
        # dℓ/dβ_j = -Σ_i (d_ij - P_ij) X_i   and   dℓ/dγ = -Σ_i Σ_j (d_ij - P_ij)(Z_ij - Z_iJ)
        P = logit_probs(θ, X, Z)
        g_β = -X' * (bigY .- P)[:, 1:J-1]
        g_γ = -sum((bigY .- P) .* (Z .- Z[:, J]))
        @test isapprox(g, [vec(g_β); g_γ])

        # negative log-likelihood of logit is convex: Hessian is positive semi-definite
        H = ForwardDiff.hessian(f, θ)
        @test isapprox(H, H', atol=1e-8)
        @test minimum(eigvals(Symmetric(H))) > -1e-8
    end

    @testset "Question 1: mlogit_with_Z agrees with GLM when J = 2" begin
        # with 2 alternatives the model is a binary logit of (y == 1) on X and Z_1 - Z_2
        Random.seed!(11)
        Nb = 1_000
        Xb = randn(Nb, K)
        Zb = randn(Nb, 2)
        θb = [0.5, -0.3, 0.8, 1.2]
        yb = draw_choices(logit_probs(θb, Xb, Zb))
        m = glm(hcat(Xb, Zb[:, 1] .- Zb[:, 2]), Float64.(yb .== 1), Binomial(), LogitLink())

        # same likelihood at the GLM estimates, and gradient is 0 there
        @test isapprox(mlogit_with_Z(coef(m), Xb, Zb, yb), -loglikelihood(m))
        g = ForwardDiff.gradient(t -> mlogit_with_Z(t, Xb, Zb, yb), coef(m))
        @test maximum(abs.(g)) < 1e-3
        # same standard errors from the Hessian
        H = ForwardDiff.hessian(t -> mlogit_with_Z(t, Xb, Zb, yb), coef(m))
        @test isapprox(sqrt.(diag(inv(H))), stderror(m), rtol=1e-4)
    end

    #---------------------------------------------------
    # Question 3
    #---------------------------------------------------
    @testset "Question 3a: practice_quadrature" begin
        @test quiet(practice_quadrature) === nothing

        # read the numbers that the function prints
        out = capture(practice_quadrature)
        integral_density = printed_numbers(out, raw"∫φ\(x\)dx =")
        expectation = printed_numbers(out, raw"∫xφ\(x\)dx =")
        @test length(integral_density) == 1
        @test length(expectation) == 1
        @test isapprox(integral_density[1], 1.0, atol=0.01)
        @test isapprox(expectation[1], 0.0, atol=1e-10)
        # value that is reported in the script
        @test isapprox(integral_density[1], 1.0045, atol=1e-4)
        @test !occursin("FILL IN", out)
    end

    @testset "Question 3b: variance_quadrature" begin
        @test quiet(variance_quadrature) === nothing

        out = capture(variance_quadrature)
        v7 = printed_numbers(out, "7 quadrature points:")
        v10 = printed_numbers(out, "10 quadrature points:")
        vtrue = printed_numbers(out, "True variance:")
        @test length(v7) == 1 && length(v10) == 1
        @test vtrue == [4.0]
        # values that are reported in the script
        @test isapprox(v7[1], 3.266, atol=1e-3)
        @test isapprox(v10[1], 4.039, atol=1e-3)
        # 10 points must be better than 7 points
        @test abs(v10[1] - 4) < abs(v7[1] - 4)
        @test isapprox(v10[1], 4.0, atol=0.05)

        # same integrals computed here
        d = Normal(0, 2)
        nodes7, weights7 = lgwt(7, -10, 10)
        nodes10, weights10 = lgwt(10, -10, 10)
        @test isapprox(v7[1], sum(weights7 .* (nodes7.^2) .* pdf.(d, nodes7)))
        @test isapprox(v10[1], sum(weights10 .* (nodes10.^2) .* pdf.(d, nodes10)))
        # with many points we get the true variance
        nodes40, weights40 = lgwt(40, -10, 10)
        @test isapprox(sum(weights40 .* (nodes40.^2) .* pdf.(d, nodes40)), 4.0, atol=1e-3)
    end

    @testset "Question 3c: practice_monte_carlo" begin
        @test quiet(practice_monte_carlo) === nothing

        # mc_integrate is defined inside practice_monte_carlo, so we read the
        # numbers that the function prints
        Random.seed!(1234)
        out = capture(practice_monte_carlo)
        D = printed_numbers(out, "With D =")
        v = printed_numbers(out, "MC Variance:")
        m = printed_numbers(out, "MC Mean:")
        dens = printed_numbers(out, "MC Density integral:")
        @test D == [1_000, 1_000_000]
        @test length(v) == 2 && length(m) == 2 && length(dens) == 2
        @test !occursin("FILL IN", out)

        # D = 1,000,000: very close to the true values 4, 0 and 1
        @test isapprox(v[2], 4.0, atol=0.05)
        @test isapprox(m[2], 0.0, atol=0.03)
        @test isapprox(dens[2], 1.0, atol=0.01)

        # D = 1,000: only roughly close
        @test isapprox(v[1], 4.0, atol=1.0)
        @test isapprox(m[1], 0.0, atol=0.6)
        @test isapprox(dens[1], 1.0, atol=0.25)

        # the function uses random numbers: other seed gives other numbers,
        # same seed gives same numbers
        Random.seed!(1234)
        out_same = capture(practice_monte_carlo)
        Random.seed!(4321)
        out_other = capture(practice_monte_carlo)
        @test out_same == out
        @test out_other != out
    end

    #---------------------------------------------------
    # Question 4
    #---------------------------------------------------
    nodes, weights = lgwt(7, -4, 4)
    # sum of ω_r φ(ξ_r): 1.0045 and not exactly 1, it gives a constant in the likelihood
    S = sum(weights .* pdf.(Normal(0, 1), nodes))
    θm = [θ; 0.7]   # [alphas; μ_γ; σ_γ], μ_γ is the γ of θ

    @testset "Question 4: mixed_logit_quad (values)" begin
        ll = mixed_logit_quad(θm, X, Z, y, nodes, weights)
        @test ll isa Float64
        @test isfinite(ll)

        # compare with equation (2) of the PS computed here:
        # for each node, logit probabilities with γ_r = μ_γ + σ_γ ξ_r, weight ω_r φ(ξ_r)
        Pint = zeros(N, J)
        for r in eachindex(nodes)
            Pint .+= weights[r] * pdf(Normal(0, 1), nodes[r]) .* logit_probs([θ[1:end-1]; θ[end] + 0.7*nodes[r]], X, Z)
        end
        @test isapprox(ll, -sum(log(Pint[n, y[n]]) for n in 1:N))
        # integrated probabilities sum to S for each person
        @test all(isapprox.(sum(Pint, dims=2), S))

        # with only one node at 0 and weight 1/φ(0), it is the multinomial logit
        @test isapprox(mixed_logit_quad(θm, X, Z, y, [0.0], [1/pdf(Normal(0, 1), 0.0)]),
                       mlogit_with_Z(θ, X, Z, y))

        # the order of the observations does not matter
        perm = randperm(N)
        @test isapprox(ll, mixed_logit_quad(θm, X[perm, :], Z[perm, :], y[perm], nodes, weights))

        # works with integer X like in the real data
        Xint = hcat(rand(18:45, N), rand(0:1, N), rand(0:1, N))
        θs = [0.05 .* θ[1:end-1]; 0.5; 0.7]
        @test isapprox(mixed_logit_quad(θs, Xint, Z, y, nodes, weights),
                       mixed_logit_quad(θs, Float64.(Xint), Z, y, nodes, weights))
    end

    @testset "Question 4: mixed_logit_quad (properties of the model)" begin
        ll = mixed_logit_quad(θm, X, Z, y, nodes, weights)

        # when σ_γ = 0, γ is constant so mixed logit is the multinomial logit.
        # The difference is the constant N*log(S)
        @test isapprox(mixed_logit_quad([θ; 0.0], X, Z, y, nodes, weights),
                       mlogit_with_Z(θ, X, Z, y) - N*log(S))

        # the sign of σ_γ does not matter (normal distribution is symmetric)
        @test isapprox(ll, mixed_logit_quad([θ; -0.7], X, Z, y, nodes, weights))

        # σ_γ and μ_γ must matter
        @test ll != mixed_logit_quad([θ; 0.0], X, Z, y, nodes, weights)
        @test ll != mixed_logit_quad([θ[1:end-1]; θ[end] + 1.0; 0.7], X, Z, y, nodes, weights)

        # if all the Z are the same for a person, μ_γ and σ_γ do not matter
        Zsame = repeat(randn(N), 1, J)
        @test isapprox(mixed_logit_quad([θ[1:end-1]; 0.0; 0.0], X, Zsame, y, nodes, weights),
                       mixed_logit_quad([θ[1:end-1]; 3.0; 2.0], X, Zsame, y, nodes, weights))

        # only differences of Z matter
        @test isapprox(ll, mixed_logit_quad(θm, X, Z .+ randn(N), y, nodes, weights))

        # the integral converges when we use more nodes (we take out the constant)
        llc(n) = begin
            nd, wt = lgwt(n, -6, 6)
            mixed_logit_quad(θm, X, Z, y, nd, wt) + N*log(sum(wt .* pdf.(Normal(0, 1), nd)))
        end
        @test isapprox(llc(30), llc(60), atol=1e-3)
        @test abs(llc(7) - llc(60)) > abs(llc(15) - llc(60))

        # Question 3d: Monte Carlo is quadrature with nodes U[a,b] and weights (b-a)/D
        Random.seed!(8)
        D = 20_000
        nd_mc = rand(D) .* 8 .- 4
        wt_mc = fill(8/D, D)
        ll_d = mixed_logit_quad(θm, X, Z, y, nd_mc, wt_mc) + N*log(sum(wt_mc .* pdf.(Normal(0, 1), nd_mc)))
        @test isapprox(ll_d, llc(60), rtol=0.01)
    end

    @testset "Question 4: mixed_logit_quad (derivatives)" begin
        f = t -> mixed_logit_quad(t, X, Z, y, nodes, weights)
        g = ForwardDiff.gradient(f, θm)
        @test length(g) == K*(J-1) + 2
        @test all(isfinite, g)
        @test isapprox(g, fd_gradient(f, θm), atol=1e-4)
        # derivative with respect to σ_γ is 0 at σ_γ = 0 (likelihood is symmetric in σ_γ)
        g0 = ForwardDiff.gradient(f, [θ; 0.0])
        @test isapprox(g0[end], 0.0, atol=1e-8)
        # at σ_γ = 0 the other derivatives are the ones of the multinomial logit
        @test isapprox(g0[1:end-1], ForwardDiff.gradient(t -> mlogit_with_Z(t, X, Z, y), θ))
    end

    #---------------------------------------------------
    # Question 5
    #---------------------------------------------------
    @testset "Question 5: mixed_logit_mc (values)" begin
        ll = mixed_logit_mc(θm, X, Z, y, 50)
        @test ll isa Float64
        @test isfinite(ll)
        @test ll > 0

        # when σ_γ = 0, all draws are equal to μ_γ so we get exactly the multinomial logit
        @test isapprox(mixed_logit_mc([θ; 0.0], X, Z, y, 10), mlogit_with_Z(θ, X, Z, y))

        # the function takes its draws from MersenneTwister(1234), we do the same here.
        # With one draw, it is the multinomial logit with γ equal to this draw
        rng = MersenneTwister(1234)
        γ_draw = rand(rng, Normal(θ[end], 0.7))
        @test isapprox(mixed_logit_mc(θm, X, Z, y, 1), mlogit_with_Z([θ[1:end-1]; γ_draw], X, Z, y))

        # with D draws, compare with the average of the logit probabilities computed here
        rng = MersenneTwister(1234)
        draws = [rand(rng, Normal(θ[end], 0.7)) for d in 1:25]
        Pavg = mean(logit_probs([θ[1:end-1]; γd], X, Z) for γd in draws)
        @test isapprox(mixed_logit_mc(θm, X, Z, y, 25), -sum(log(Pavg[n, y[n]]) for n in 1:N))
        # simulated probabilities sum to 1 for each person
        @test all(isapprox.(sum(Pavg, dims=2), 1.0))

        # if all the Z are the same for a person, the draws do not matter
        Zsame = repeat(randn(N), 1, J)
        @test isapprox(mixed_logit_mc(θm, X, Zsame, y, 20), mlogit_with_Z(θ, X, Zsame, y))
    end

    @testset "Question 5: mixed_logit_mc (random draws)" begin
        # the function uses the same draws each time we call it (if not, the
        # optimizer cannot converge): two calls give exactly the same value
        ll1 = mixed_logit_mc(θm, X, Z, y, 50)
        @test mixed_logit_mc(θm, X, Z, y, 50) == ll1

        # ... and the seed of the global random numbers does not change it
        Random.seed!(5); ll2 = mixed_logit_mc(θm, X, Z, y, 50)
        Random.seed!(6); ll3 = mixed_logit_mc(θm, X, Z, y, 50)
        @test ll2 == ll1
        @test ll3 == ll1

        # ... and the function does not use the global random numbers
        Random.seed!(5); r1 = rand()
        Random.seed!(5); mixed_logit_mc(θm, X, Z, y, 50); r2 = rand()
        @test r1 == r2

        # the number of draws matters
        @test mixed_logit_mc(θm, X, Z, y, 51) != ll1

        # negative σ_γ must work (the optimizer can try it) and give the same as positive σ_γ
        @test mixed_logit_mc([θ; -0.7], X, Z, y, 50) == ll1

        # with many draws, Monte Carlo and quadrature give almost the same likelihood
        # (we use many quadrature points and take out the constant of the weights)
        nodes40, weights40 = lgwt(40, -6, 6)
        S40 = sum(weights40 .* pdf.(Normal(0, 1), nodes40))
        ll_quad = mixed_logit_quad(θm, X, Z, y, nodes40, weights40) + N*log(S40)
        @test isapprox(mixed_logit_mc(θm, X, Z, y, 5_000), ll_quad, rtol=0.01)

        # simulation error is smaller when we use more draws
        @test abs(mixed_logit_mc(θm, X, Z, y, 5_000) - ll_quad) < abs(mixed_logit_mc(θm, X, Z, y, 20) - ll_quad)
    end

    @testset "Question 5: mixed_logit_mc (derivatives)" begin
        # automatic differentiation works, also with negative σ_γ
        for σ in (0.7, -0.7)
            g = ForwardDiff.gradient(t -> mixed_logit_mc(t, X, Z, y, 10), [θ; σ])
            @test length(g) == K*(J-1) + 2
            @test all(isfinite, g)
        end
        # gradient agree with finite differences (possible because the draws are fixed)
        f = t -> mixed_logit_mc(t, X, Z, y, 10)
        @test isapprox(ForwardDiff.gradient(f, θm), fd_gradient(f, θm), atol=1e-4)
        # at σ_γ = 0 the derivatives for alphas and μ_γ are the ones of the multinomial logit
        g0 = ForwardDiff.gradient(f, [θ; 0.0])
        @test isapprox(g0[1:end-1], ForwardDiff.gradient(t -> mlogit_with_Z(t, X, Z, y), θ))
    end

    #---------------------------------------------------
    # Estimation
    #---------------------------------------------------

    # with simulated data, estimates should be close to true values
    # (K = 3 and J = 8 like the real data, because starting values are fixed in the function)
    @testset "Question 1: optimize_mlogit recovers parameters" begin
        Random.seed!(2)
        Ns = 10_000
        Xs = randn(Ns, K)
        Zs = randn(Ns, J)
        θtrue = [0.5 .* randn(K*(J-1)); 0.8]
        ys = draw_choices(logit_probs(θtrue, Xs, Zs))
        f = t -> mlogit_with_Z(t, Xs, Zs, ys)

        θ̂, se = quiet(() -> optimize_mlogit(Xs, Zs, ys))
        @test length(θ̂) == K*(J-1) + 1
        @test length(se) == K*(J-1) + 1
        @test all(se .> 0)
        @test isapprox(θ̂, θtrue, atol=0.2)
        @test isapprox(θ̂[end], θtrue[end], atol=0.1)

        # estimates are not far from true values compared to the standard errors
        @test all(abs.(θ̂ .- θtrue) .< 4 .* se)

        # it is a minimum: gradient is 0 and likelihood is lower than at true values
        @test maximum(abs.(ForwardDiff.gradient(f, θ̂))) < 1e-3
        @test f(θ̂) <= f(θtrue)

        # standard errors are the ones of the inverse Hessian
        @test isapprox(se, sqrt.(diag(inv(ForwardDiff.hessian(f, θ̂)))), rtol=1e-6)

        # starting values are fixed, so we get the same result if we run again
        θ̂2, se2 = quiet(() -> optimize_mlogit(Xs, Zs, ys))
        @test θ̂2 == θ̂
        @test se2 == se

        # with 4 times less observations, standard errors are about 2 times bigger
        q = 1:2_500
        θ̂q, seq = quiet(() -> optimize_mlogit(Xs[q, :], Zs[q, :], ys[q]))
        @test 1.5 < seq[end] / se[end] < 2.5
        @test all(1.5 .< seq ./ se .< 2.5)
    end

    # The two mixed logit optimizers are not run on the real data (too slow).
    # Here we run the quadrature one on small simulated data sets.
    @testset "Question 4: optimize_mixed_logit_quad on small data" begin
        Random.seed!(3)
        Ns = 500
        Xs = randn(Ns, K)
        Zs = randn(Ns, J)
        θtrue = [0.5 .* randn(K*(J-1)); 0.8]
        ys = draw_choices(logit_probs(θtrue, Xs, Zs))

        # starting values are the multinomial logit estimates
        θ̂mnl, _ = quiet(() -> optimize_mlogit(Xs, Zs, ys))
        θ̂ = quiet(() -> optimize_mixed_logit_quad(Xs, Zs, ys, θ̂mnl))
        @test length(θ̂) == K*(J-1) + 2
        @test all(isfinite, θ̂)
        # likelihood at the estimates is better than at 0 and than at the starting values
        fq = t -> mixed_logit_quad(t, Xs, Zs, ys, nodes, weights)
        @test fq(θ̂) < fq([zeros(K*(J-1)); 0.1; 1.0])
        @test fq(θ̂) <= fq([θ̂mnl; 1.0])

        # no random starting values now: same result if we run again, and the
        # function does not use the random numbers
        Random.seed!(5); r1 = rand()
        Random.seed!(5); θ̂2 = quiet(() -> optimize_mixed_logit_quad(Xs, Zs, ys, θ̂mnl)); r2 = rand()
        @test θ̂2 == θ̂
        @test r1 == r2

        # the starting values are really used: with only 1 parameter instead
        # of 22 the function gives an error
        @test_throws Exception quiet(() -> optimize_mixed_logit_quad(Xs, Zs, ys, [0.0]))
    end

    @testset "Question 4: optimize_mixed_logit_quad recovers parameters" begin
        # data from a true mixed logit: each person has his own γ_i ~ N(0.8, 1)
        Random.seed!(3)
        Ns = 5_000
        Xs = randn(Ns, K)
        Zs = randn(Ns, J)
        αtrue = 0.5 .* randn(K*(J-1))
        μtrue, σtrue = 0.8, 1.0
        γi = μtrue .+ σtrue .* randn(Ns)
        v = Xs * [reshape(αtrue, K, J-1) zeros(K)] .+ γi .* (Zs .- Zs[:, J])
        ys = draw_choices(exp.(v) ./ sum(exp.(v), dims=2))
        f = t -> mixed_logit_quad(t, Xs, Zs, ys, nodes, weights)

        θ̂mnl, _ = quiet(() -> optimize_mlogit(Xs, Zs, ys))
        θ̂ = quiet(() -> optimize_mixed_logit_quad(Xs, Zs, ys, θ̂mnl))
        @test maximum(abs.(θ̂[1:end-2] .- αtrue)) < 0.25
        @test isapprox(θ̂[end-1], μtrue, atol=0.15)
        @test isapprox(abs(θ̂[end]), σtrue, atol=0.3)   # sign of σ_γ is not identified

        # it is a minimum: gradient is 0 and likelihood is lower than at true values
        @test maximum(abs.(ForwardDiff.gradient(f, θ̂))) < 1e-2
        @test f(θ̂) <= f([αtrue; μtrue; σtrue])

        # mixed logit fits better than multinomial logit (σ_γ = 0) on these data
        @test f(θ̂) < f([θ̂mnl; 0.0])
    end

    # This one is slow (about 1 minute): 1,000 draws for each evaluation of the likelihood
    @testset "Question 5: optimize_mixed_logit_mc recovers parameters" begin
        # data from a true mixed logit: each person has his own γ_i ~ N(0.8, 1)
        Random.seed!(3)
        Ns = 1_000
        Xs = randn(Ns, K)
        Zs = randn(Ns, J)
        αtrue = 0.5 .* randn(K*(J-1))
        μtrue, σtrue = 0.8, 1.0
        γi = μtrue .+ σtrue .* randn(Ns)
        v = Xs * [reshape(αtrue, K, J-1) zeros(K)] .+ γi .* (Zs .- Zs[:, J])
        ys = draw_choices(exp.(v) ./ sum(exp.(v), dims=2))
        f = t -> mixed_logit_mc(t, Xs, Zs, ys, 1_000)   # D = 1,000 like in the function

        θ̂mnl, _ = quiet(() -> optimize_mlogit(Xs, Zs, ys))
        θ̂ = quiet(() -> optimize_mixed_logit_mc(Xs, Zs, ys, θ̂mnl))
        @test length(θ̂) == K*(J-1) + 2
        @test all(isfinite, θ̂)
        # tolerances are bigger than for quadrature because we have only 1,000 observations
        @test maximum(abs.(θ̂[1:end-2] .- αtrue)) < 0.5
        @test isapprox(θ̂[end-1], μtrue, atol=0.25)
        @test isapprox(abs(θ̂[end]), σtrue, atol=0.35)   # sign of σ_γ is not identified

        # it is a minimum: gradient is 0 and likelihood is lower than at true values
        @test maximum(abs.(ForwardDiff.gradient(f, θ̂))) < 1e-3
        @test f(θ̂) <= f([αtrue; μtrue; σtrue])

        # quadrature and Monte Carlo estimates are close on the same data
        θ̂quad = quiet(() -> optimize_mixed_logit_quad(Xs, Zs, ys, θ̂mnl))
        @test maximum(abs.(θ̂[1:end-2] .- θ̂quad[1:end-2])) < 0.1
        @test isapprox(θ̂[end-1], θ̂quad[end-1], atol=0.1)
        @test isapprox(abs(θ̂[end]), abs(θ̂quad[end]), atol=0.2)
    end

    # allwrap is not tested: it runs all the problem set (about 6 minutes).
    # The tests below check its results on the real data without running it.

    #---------------------------------------------------
    # Real data
    #---------------------------------------------------
    @testset "load_data and results on the real data" begin
        df, Xr, Zr, yr = load_data()
        @test size(Xr) == (28365, 3)
        @test size(Zr) == (28365, 8)
        @test length(yr) == 28365
        @test sort(unique(yr)) == collect(1:8)
        @test Xr == [df.age df.white df.collgrad]
        @test yr == df.occ_code
        @test Zr[:, 8] == df.elnwage8
        @test all(x -> x in (0, 1), Xr[:, 2:3])    # white and collgrad are dummies
        @test !any(ismissing, Xr) && !any(ismissing, Zr) && !any(ismissing, yr)
        # the file in the folder is the same data
        @test load_data()[3] == Matrix(CSV.read("nlsw88t.csv", DataFrame)[:, r"elnwage"])

        # estimates and standard errors that allwrap() prints for Question 1
        θ̂ = [0.04037451851180445, 0.24399384763664328, -1.5713223577272584,
             0.043325554704952064, 0.14685399864279708, -2.959104412663344,
             0.10205746878371494, 0.747307531242044, -4.120051573371816,
             0.03756289315341755, 0.6884881358453606, -3.6557710485690187,
             0.020454395106744842, -0.35840345045460625, -4.376932265855823,
             0.10746370393233136, -0.5263752811064953, -6.19919267037392,
             0.11688246835750415, -0.2870564175012443, -5.322249316115956,
             1.3074736854119615]
        # starting values of optimize_mlogit (estimates of PS3)
        θps3 = [0.0557, 0.0834, -2.3449, 0.0450, 0.7366, -3.1532, 0.0926, -0.0842, -4.2733,
                0.0239, 0.7231, -3.7494, 0.0361, -0.6438, -4.2797, 0.0853, -1.1714, -6.6787,
                0.0866, -0.7979, -4.9691, -0.0942]
        f = t -> mlogit_with_Z(t, Xr, Zr, yr)

        # they are a minimum of the likelihood, and better than PS3 values
        @test isapprox(f(θ̂), 44883.168, atol=0.01)
        @test f(θ̂) < f(θps3)
        @test f(θ̂) < f(zeros(22))
        # the optimizer stopped with a gradient of 0.001 (not under g_tol = 1e-5),
        # so we check with one Newton step H⁻¹g that the true minimum is very
        # close: the step must be much smaller than the standard errors
        g = ForwardDiff.gradient(f, θ̂)
        H = ForwardDiff.hessian(f, θ̂)
        @test maximum(abs.(g)) < 1e-2
        @test minimum(eigvals(Symmetric(H))) > 0     # Hessian is positive definite

        # standard errors from the Hessian, as reported in the script (Question 2)
        se = sqrt.(diag(inv(H)))
        @test all(se .> 0)
        @test maximum(abs.(H \ g) ./ se) < 1e-3
        @test isapprox(θ̂[end], 1.307, atol=1e-3)
        @test isapprox(se[end], 0.125, atol=1e-3)
        @test θ̂[end] / se[end] > 1.96       # γ̂ is positive and significant
        @test isapprox(θ̂[end] / se[end], 10.5, atol=0.05)

        # mixed logit likelihoods can be evaluated on the real data (one time, not optimized).
        # With σ_γ = 0 they are the multinomial logit likelihood
        @test isapprox(mixed_logit_quad([θ̂; 0.0], Xr, Zr, yr, nodes, weights), f(θ̂) - 28365*log(S))
        @test isapprox(mixed_logit_mc([θ̂; 0.0], Xr, Zr, yr, 3), f(θ̂))
    end
end
