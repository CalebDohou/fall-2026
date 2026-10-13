#---------------------------------------------------
# ECON 6343: Econometrics III — Problem Set 5 (Bus Engine Replacement, Rust 1987)
# Caleb Dohou
#
# Question 4: unit tests for every function in PS5_Dohou_source.jl.
# The source code is read in first, then the tests run.
#
# Run with:  julia PS5_Dohou_tests.jl
#
# Test design
#   * Each function gets its own @testset, in the order of the problem set.
#   * Slow checks (literal re-implementation, optimization, Monte Carlo) run
#     on a trimmed copy of the data (fewer buses and periods, same grid).
#   * Where possible a result is checked against an independent computation:
#     a hand-written logit MLE for GLM, a literal transcription of the PDF
#     loops for the recursion, the β = 0 limit for the likelihood, and a
#     Monte Carlo simulation with known θ for the estimator.
#---------------------------------------------------

using Test
using Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM

cd(@__DIR__)
include("create_grids.jl")
include("PS5_Dohou_source.jl")

################################################################################
# Fixtures and helpers
################################################################################

const df_long = load_static_data()
const d       = load_dynamic_data()

# A trimmed copy of the dynamic data: fewer buses and periods, same grid.
function shrink(d; N = 100, T = 4)
    return merge(d, (Y = d.Y[1:N, 1:T], X = d.X[1:N, 1:T], B = d.B[1:N],
                     Xstate = d.Xstate[1:N, 1:T], Zstate = d.Zstate[1:N],
                     N = N, T = T))
end
const dsmall = shrink(d)

# Literal transcription of PDF 3(c)-3(d): no hoisting, explicit P1 and P0.
function fv_literal(θ, d)
    FV = zeros(d.zbin * d.xbin, 2, d.T + 1)
    for t in d.T:-1:1, b in 0:1, z in 1:d.zbin, x in 1:d.xbin
        row = x + (z - 1) * d.xbin
        v1 = θ[1] + θ[2] * d.xval[x] + θ[3] * b +
             d.xtran[row, :]' * FV[(z-1)*d.xbin+1:z*d.xbin, b+1, t+1]
        v0 = d.xtran[1+(z-1)*d.xbin, :]' * FV[(z-1)*d.xbin+1:z*d.xbin, b+1, t+1]
        FV[row, b+1, t] = d.β * log(exp(v0) + exp(v1))
    end
    return FV
end

# Choice probabilities P(Y=1) for every observation, from the PDF formulas.
function choice_probs(θ, d)
    FV = fv_literal(θ, d)
    P  = zeros(d.N, d.T)
    for i in 1:d.N, t in 1:d.T
        row0 = 1 + (d.Zstate[i] - 1) * d.xbin
        row1 = d.Xstate[i, t] + (d.Zstate[i] - 1) * d.xbin
        vdiff = θ[1] + θ[2] * d.X[i, t] + θ[3] * d.B[i] +
                d.β * ((d.xtran[row1, :] .- d.xtran[row0, :])' * FV[row0:row0+d.xbin-1, d.B[i]+1, t+1])
        P[i, t] = exp(vdiff) / (1 + exp(vdiff))
    end
    return P
end

function loglike_literal(θ, d)
    P = choice_probs(θ, d)
    return sum(d.Y .* log.(P) .+ (1 .- d.Y) .* log.(1 .- P))
end

# Simulate a panel from the dynamic model with known θ, using the same grid
# and transition matrix. Mileage after a replacement is drawn from the
# f0 row of xtran, exactly as the likelihood assumes.
function simulate_panel(θ, d; N = 500, T = 10, rng = MersenneTwister(2025))
    dsim = merge(d, (N = N, T = T))
    FV = zeros(d.zbin * d.xbin, 2, T + 1)
    compute_future_value!(FV, θ, dsim)
    Y = zeros(Int, N, T); X = zeros(N, T); Xst = zeros(Int, N, T)
    Z = rand(rng, 1:d.zbin, N); B = rand(rng, 0:1, N)
    for i in 1:N
        x = 1                                   # every bus starts with a new engine
        row0 = 1 + (Z[i] - 1) * d.xbin
        for t in 1:T
            row1 = x + (Z[i] - 1) * d.xbin
            vdiff = θ[1] + θ[2] * d.xval[x] + θ[3] * B[i] +
                    d.β * ((d.xtran[row1, :] .- d.xtran[row0, :])' * FV[row0:row0+d.xbin-1, B[i]+1, t+1])
            p1 = 1 / (1 + exp(-vdiff))
            y  = rand(rng) < p1 ? 1 : 0
            Y[i, t] = y; X[i, t] = d.xval[x]; Xst[i, t] = x
            # next mileage bin: continue from x if kept, from 0 if replaced
            probs = y == 1 ? d.xtran[row1, :] : d.xtran[row0, :]
            u = rand(rng); c = 0.0; nx = d.xbin
            for j in 1:d.xbin
                c += probs[j]
                if u <= c; nx = j; break; end
            end
            x = nx
        end
    end
    return merge(dsim, (Y = Y, X = X, B = B, Xstate = Xst, Zstate = Z))
end

const θtest = [2.0, -0.15, 1.0]
const θhat  = [1.9464, -0.1642, 1.0739]     # reported dynamic optimum (full data)
const θstat = [1.926, -0.148, 1.059]        # reported static estimates

################################################################################
@testset "PS5 Dohou: all tests" begin
################################################################################

#---------------------------------------------------
# Question 1: load_static_data
#---------------------------------------------------
@testset "load_static_data (Q1)" begin
    @testset "shape and columns" begin
        @test df_long isa DataFrame
        @test nrow(df_long) == 1000 * 20
        for c in [:bus_id, :time, :Y, :Odometer, :RouteUsage, :Branded]
            @test c in propertynames(df_long)
        end
        @test length(unique(df_long.bus_id)) == 1000
        @test sort(unique(df_long.time)) == collect(1.0:20.0)
        @test all(combine(groupby(df_long, :bus_id), nrow).nrow .== 20)
    end
    @testset "values" begin
        @test all(in.(df_long.Y, Ref((0, 1))))
        @test all(in.(df_long.Branded, Ref((0, 1))))
        @test !any(ismissing, df_long.Odometer)
        @test all(0 .<= df_long.Odometer .<= 25)
        @test all(0.25 .<= df_long.RouteUsage .<= 1.25)
        @test 0.5 < mean(df_long.Y) < 1.0         # buses are mostly kept
    end
    @testset "panel structure" begin
        @test issorted(collect(zip(df_long.bus_id, df_long.time)))
        g = combine(groupby(df_long, :bus_id),
                    :Branded    => (x -> length(unique(x))) => :nb,
                    :RouteUsage => (x -> length(unique(x))) => :nr)
        @test all(g.nb .== 1) && all(g.nr .== 1)   # bus-level variables constant within bus
    end
    @testset "reshape reproduces the wide file" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
        dfw = CSV.read(HTTP.get(url).body, DataFrame)
        for i in [1, 2, 500, 1000], t in [1, 7, 20]
            r = df_long[(df_long.bus_id .== i) .& (df_long.time .== t), :]
            @test nrow(r) == 1
            @test r.Y[1]        == dfw[i, Symbol("Y$t")]
            @test r.Odometer[1] == dfw[i, Symbol("Odo$t")]
            @test r.Branded[1]  == dfw[i, :Branded]
        end
    end
end

#---------------------------------------------------
# Question 3b: create_grids
#---------------------------------------------------
@testset "create_grids (Q3b)" begin
    zval, zbin, xval, xbin, xtran = create_grids()
    @testset "grids" begin
        @test xval[1] == 0.0 && xval[end] == 25.0 && xbin == 201
        @test all(diff(xval) .≈ 0.125)
        @test zval[1] ≈ 0.25 && zval[end] ≈ 1.25 && zbin == 101
        @test all(diff(zval) .≈ 0.01)
    end
    @testset "Markov matrix properties" begin
        @test size(xtran) == (zbin * xbin, xbin)
        @test all(xtran .>= 0)
        @test all(abs.(sum(xtran, dims = 2) .- 1) .< 1e-12)
        # no mass below the current mileage bin (mileage never decreases)
        for z in [1, 50, 101], x in [1, 100, 201]
            row = x + (z - 1) * xbin
            @test all(xtran[row, 1:x-1] .== 0)
            @test xtran[row, x] > 0
        end
        # from the top bin the bus stays in the top bin with certainty
        @test xtran[xbin, xbin] ≈ 1 && xtran[zbin * xbin, xbin] ≈ 1
    end
    @testset "matches equation (3) column by column" begin
        for z in [1, 37, 101], x in [1, 20, 150]
            row = x + (z - 1) * xbin
            for j in x:xbin-1          # interior columns use the exact formula
                expected = exp(-zval[z] * (xval[j] - xval[x])) - exp(-zval[z] * (xval[j] + 0.125 - xval[x]))
                @test xtran[row, j] ≈ expected atol = 1e-12
            end
            # last column absorbs the tail of the exponential
            @test xtran[row, xbin] ≈ exp(-zval[z] * (xval[xbin] - xval[x])) atol = 1e-12
        end
    end
    @testset "economic content" begin
        # transitions depend only on the mileage increment: row for x is row for 1 shifted
        for z in [10, 90]
            r1 = 1 + (z - 1) * xbin; r5 = 5 + (z - 1) * xbin
            @test xtran[r5, 5:xbin-1] ≈ xtran[r1, 1:xbin-5]
        end
        # higher x2 means lower usage intensity: more mass on small increments
        p_stay_low  = xtran[1, 1]
        p_stay_high = xtran[1 + (zbin - 1) * xbin, 1]
        @test p_stay_high > p_stay_low
        @test p_stay_low ≈ 1 - exp(-0.25 * 0.125)
        # expected mileage increment is decreasing in x2
        ex(z) = xtran[1 + (z - 1) * xbin, :]' * xval
        @test ex(1) > ex(50) > ex(101)
    end
end

#---------------------------------------------------
# Question 3a: load_dynamic_data
#---------------------------------------------------
@testset "load_dynamic_data (Q3a)" begin
    @testset "dimensions and types" begin
        @test d.N == 1000 && d.T == 20
        @test size(d.Y) == (1000, 20) && size(d.X) == (1000, 20) && size(d.Xstate) == (1000, 20)
        @test length(d.Zstate) == 1000 && length(d.B) == 1000
        @test eltype(d.Y) <: Integer && eltype(d.Xstate) <: Integer && eltype(d.Zstate) <: Integer
        @test eltype(d.X) <: AbstractFloat
        @test d.β == 0.9
        @test d.xbin == 201 && d.zbin == 101
        @test size(d.xtran) == (d.zbin * d.xbin, d.xbin)
        @test length(d.xval) == d.xbin && length(d.zval) == d.zbin
    end
    @testset "values" begin
        @test all(in.(d.Y, Ref((0, 1))))
        @test all(in.(d.B, Ref((0, 1))))
        @test 1 <= minimum(d.Xstate) && maximum(d.Xstate) <= d.xbin
        @test 1 <= minimum(d.Zstate) && maximum(d.Zstate) <= d.zbin
        @test all(d.xval[d.Xstate] .≈ d.X)            # bin index reproduces the odometer value
        @test 0.5 < mean(d.Y) < 1.0
    end
    @testset "dynamics in the data" begin
        # mileage never falls while the engine is kept
        kept = d.Y[:, 1:end-1] .== 1
        @test all((d.Xstate[:, 2:end] .- d.Xstate[:, 1:end-1])[kept] .>= 0)
        # mileage restarts from the bottom after every replacement
        repl = d.Y[:, 1:end-1] .== 0
        @test all(d.Xstate[:, 2:end][repl] .<= 2)
        @test any(repl)
    end
    @testset "consistent with the static data file" begin
        # same buses in both files: brand and route usage agree bus by bus
        first_rows = combine(groupby(df_long, :bus_id), :Branded => first => :B, :RouteUsage => first => :RU)
        @test first_rows.B == d.B
        @test all(first_rows.RU .≈ d.zval[d.Zstate])
    end
end

#---------------------------------------------------
# Question 2: estimate_static_model
#---------------------------------------------------
@testset "estimate_static_model (Q2)" begin
    m  = estimate_static_model(df_long)
    θs = coef(m)
    @testset "basic output" begin
        @test length(θs) == 3
        @test all(isfinite, θs)
        @test θs[1] > 0 && θs[2] < 0 && θs[3] > 0
        @test all(stderror(m) .> 0)
        @test all(abs.(θs ./ stderror(m)) .> 10)     # all highly significant
        @test θs ≈ θstat atol = 1e-2                 # regression check on this data
    end
    @testset "agrees with an independent hand-written logit MLE" begin
        Xmat = hcat(ones(nrow(df_long)), df_long.Odometer, df_long.Branded)
        y    = df_long.Y
        nll(θ) = -sum(y .* (Xmat * θ) .- log.(1 .+ exp.(Xmat * θ)))
        res = optimize(nll, zeros(3), LBFGS(), Optim.Options(g_tol = 1e-8))
        @test res.minimizer ≈ θs atol = 1e-4
        @test res.minimum ≈ deviance(m) / 2 atol = 1e-6
    end
    @testset "logit score equations hold at the MLE" begin
        p = predict(m)
        @test all(0 .< p .< 1)
        # sums over 20,000 observations; GLM's IRLS stops at ~1e-3, i.e. < 1e-7 per observation
        @test sum(df_long.Y .- p) ≈ 0 atol = 1e-2                        # intercept
        @test sum((df_long.Y .- p) .* df_long.Odometer) ≈ 0 atol = 1e-2  # mileage
        @test sum((df_long.Y .- p) .* df_long.Branded) ≈ 0 atol = 1e-2   # brand
        # predicted keep-probability falls with mileage
        @test cor(p, df_long.Odometer) < 0
    end
    @testset "works on a subset and on a copy" begin
        sub = df_long[df_long.bus_id .<= 200, :]
        ms  = estimate_static_model(sub)
        @test length(coef(ms)) == 3 && all(isfinite, coef(ms))
        @test coef(estimate_static_model(copy(df_long))) == θs
    end
end

#---------------------------------------------------
# Question 3c: compute_future_value!
#---------------------------------------------------
@testset "compute_future_value! (Q3c)" begin
    FV = zeros(d.zbin * d.xbin, 2, d.T + 1)
    compute_future_value!(FV, θtest, d)
    @testset "shape, terminal condition, finiteness" begin
        @test size(FV) == (d.zbin * d.xbin, 2, d.T + 1)
        @test all(FV[:, :, end] .== 0)
        @test all(FV[:, :, 1:end-1] .> 0)
        @test all(isfinite, FV)
    end
    @testset "last period has a closed form" begin
        # v0 = 0 and v1 = flow utility, so FV_T = β·log(1 + exp(u1)) exactly
        for z in [1, 50, 101], x in [1, 100, 201], b in 0:1
            row = x + (z - 1) * d.xbin
            u1 = θtest[1] + θtest[2] * d.xval[x] + θtest[3] * b
            @test FV[row, b+1, d.T] ≈ d.β * log(1 + exp(u1))
        end
    end
    @testset "monotonicity in states and parameters" begin
        blk(z) = ((z - 1) * d.xbin + 1):(z * d.xbin)
        @test issorted(FV[blk(50), 1, 1], rev = true)       # falls with mileage (θ1 < 0)
        @test all(FV[:, 2, :] .>= FV[:, 1, :])              # branded worth more (θ2 > 0)
        @test all(FV[:, :, 1] .>= FV[:, :, d.T])            # more periods ahead, more value
        # higher usage intensity (low x2) means faster mileage growth, lower value
        @test FV[1, 1, 1] < FV[1 + 100 * d.xbin, 1, 1]
        # raising θ0 raises every future value
        FVhi = zeros(size(FV)); compute_future_value!(FVhi, θtest .+ [1, 0, 0], d)
        @test all(FVhi[:, :, 1:end-1] .> FV[:, :, 1:end-1])
        # a higher discount factor raises every future value
        FVlo = zeros(size(FV)); compute_future_value!(FVlo, θtest, merge(d, (β = 0.5,)))
        @test all(FVlo[:, :, 1:end-1] .< FV[:, :, 1:end-1])
        # with β = 0 every future value is zero
        FV0 = zeros(size(FV)); compute_future_value!(FV0, θtest, merge(d, (β = 0.0,)))
        @test all(FV0 .== 0)
    end
    @testset "independent of the observed data" begin
        FVs = zeros(dsmall.zbin * dsmall.xbin, 2, d.T + 1)
        compute_future_value!(FVs, θtest, merge(dsmall, (T = d.T,)))
        @test FVs == FV
    end
    @testset "matches a literal transcription of the PDF loops" begin
        FVs = zeros(dsmall.zbin * dsmall.xbin, 2, dsmall.T + 1)
        compute_future_value!(FVs, θtest, dsmall)
        @test maximum(abs.(FVs .- fv_literal(θtest, dsmall))) < 1e-10
    end
    @testset "edge cases and purity" begin
        # T = 1: one slice filled, terminal slice zero
        d1 = merge(dsmall, (T = 1,))
        FV1 = zeros(d.zbin * d.xbin, 2, 2); compute_future_value!(FV1, θtest, d1)
        @test all(FV1[:, :, 2] .== 0) && all(FV1[:, :, 1] .> 0)
        # overwrites whatever was in FV (does not accumulate)
        FVg = fill(123.0, size(FV1)); FVg[:, :, end] .= 0
        compute_future_value!(FVg, θtest, d1)
        @test FVg == FV1
        # integer θ and extreme θ are handled without NaN/Inf
        FVi = zeros(size(FV1)); compute_future_value!(FVi, [2, 0, 1], d1)
        @test all(isfinite, FVi)
        FVx = zeros(size(FV1)); compute_future_value!(FVx, [10.0, -2.0, 5.0], d1)
        @test all(isfinite, FVx)
        # inputs are not mutated
        xt = copy(d.xtran); xv = copy(d.xval)
        compute_future_value!(zeros(size(FV1)), θtest, d1)
        @test d.xtran == xt && d.xval == xv
        @test compute_future_value!(zeros(size(FV1)), θtest, d1) === nothing
    end
end

#---------------------------------------------------
# Question 3d-3f: log_likelihood_dynamic
#---------------------------------------------------
@testset "log_likelihood_dynamic (Q3d-3f)" begin
    nll = log_likelihood_dynamic(θtest, d)
    @testset "basic output" begin
        @test nll isa Float64
        @test isfinite(nll) && nll > 0
        @test nll < 2 * d.N * d.T * log(2)
        @test log_likelihood_dynamic(θtest, d) == nll          # deterministic, fresh FV each call
        @test log_likelihood_dynamic(θhat, d) ≈ 8765.84 atol = 0.01   # reported optimum
        @test @inferred(log_likelihood_dynamic(θtest, dsmall)) isa Float64
    end
    @testset "matches the literal PDF formulas" begin
        @test log_likelihood_dynamic(θtest, dsmall) ≈ -loglike_literal(θtest, dsmall) atol = 1e-8
        @test log_likelihood_dynamic(θhat,  dsmall) ≈ -loglike_literal(θhat,  dsmall) atol = 1e-8
        P = choice_probs(θtest, dsmall)
        @test all(0 .< P .< 1)
    end
    @testset "β = 0 collapses to the static logit" begin
        d0  = merge(d, (β = 0.0,))
        dfl = DataFrame(Y = vec(d.Y), Odometer = vec(d.X), Branded = repeat(d.B, d.T))
        g   = glm(@formula(Y ~ Odometer + Branded), dfl, Binomial(), LogitLink())
        @test log_likelihood_dynamic(coef(g), d0) ≈ deviance(g) / 2 atol = 1e-6
        @test log_likelihood_dynamic(zeros(3), d0) ≈ d.N * d.T * log(2) atol = 1e-8
        # gradient of the β = 0 likelihood vanishes at the GLM coefficients
        h = 1e-5
        g0 = [(log_likelihood_dynamic(coef(g) .+ h .* (1:3 .== k), d0) -
               log_likelihood_dynamic(coef(g) .- h .* (1:3 .== k), d0)) / (2h) for k in 1:3]
        @test maximum(abs.(g0)) < 1e-3
    end
    @testset "algebraic identities" begin
        # additive across buses
        dA = shrink(d, N = 60, T = 4); dB = merge(dsmall, (Y = d.Y[61:100, 1:4], X = d.X[61:100, 1:4],
                   B = d.B[61:100], Xstate = d.Xstate[61:100, 1:4], Zstate = d.Zstate[61:100], N = 40))
        @test log_likelihood_dynamic(θtest, dA) + log_likelihood_dynamic(θtest, dB) ≈
              log_likelihood_dynamic(θtest, dsmall) atol = 1e-8
        # invariant to the order of buses
        perm = randperm(MersenneTwister(7), dsmall.N)
        dP = merge(dsmall, (Y = dsmall.Y[perm, :], X = dsmall.X[perm, :], B = dsmall.B[perm],
                            Xstate = dsmall.Xstate[perm, :], Zstate = dsmall.Zstate[perm]))
        @test log_likelihood_dynamic(θtest, dP) ≈ log_likelihood_dynamic(θtest, dsmall) atol = 1e-8
        # nll(Y) + nll(1-Y) = -Σ[log P1 + log P0], the same for any Y
        d1s = merge(dsmall, (Y = ones(Int, dsmall.N, dsmall.T),))
        d0s = merge(dsmall, (Y = zeros(Int, dsmall.N, dsmall.T),))
        dcs = merge(dsmall, (Y = 1 .- dsmall.Y,))
        @test log_likelihood_dynamic(θtest, d1s) + log_likelihood_dynamic(θtest, d0s) ≈
              log_likelihood_dynamic(θtest, dsmall) + log_likelihood_dynamic(θtest, dcs) atol = 1e-8
        # a bus that is always kept is more likely under a larger θ0
        @test log_likelihood_dynamic(θtest .+ [1, 0, 0], d1s) < log_likelihood_dynamic(θtest, d1s)
    end
    @testset "shape of the objective" begin
        @test log_likelihood_dynamic(θhat, d) < log_likelihood_dynamic(θstat, d)
        for δ in ([0.3, 0, 0], [0, 0.03, 0], [0, 0, 0.3])
            @test log_likelihood_dynamic(θhat .+ δ, d) > log_likelihood_dynamic(θhat, d)
            @test log_likelihood_dynamic(θhat .- δ, d) > log_likelihood_dynamic(θhat, d)
        end
    end
    @testset "robustness and purity" begin
        @test isfinite(log_likelihood_dynamic([2, 0, 1], dsmall))           # integer θ
        @test isfinite(log_likelihood_dynamic([10.0, -2.0, 5.0], dsmall))   # extreme θ
        @test isfinite(log_likelihood_dynamic([-5.0, 0.5, -3.0], dsmall))   # wrong-signed θ
        Y0 = copy(d.Y); X0 = copy(d.X); xt = copy(d.xtran)
        log_likelihood_dynamic(θtest, dsmall)
        @test d.Y == Y0 && d.X == X0 && d.xtran == xt
    end
end

#---------------------------------------------------
# Question 3e-3h: estimate_dynamic_model
#---------------------------------------------------
@testset "estimate_dynamic_model (Q3e-3h)" begin
    res = redirect_stdout(devnull) do
        estimate_dynamic_model(dsmall, θ_start = θtest)
    end
    @testset "optimizer output" begin
        @test res isa Optim.MultivariateOptimizationResults
        @test Optim.converged(res)
        @test length(res.minimizer) == 3 && all(isfinite, res.minimizer)
        @test res.minimum <= log_likelihood_dynamic(θtest, dsmall)
        @test res.minimum ≈ log_likelihood_dynamic(res.minimizer, dsmall)
        @test Optim.iterations(res) >= 1
        # numerically zero gradient at the optimum
        h = 1e-5
        f(θ) = log_likelihood_dynamic(θ, dsmall)
        grad = [(f(res.minimizer .+ h .* (1:3 .== k)) - f(res.minimizer .- h .* (1:3 .== k))) / (2h) for k in 1:3]
        @test maximum(abs.(grad)) < 1e-2
    end
    @testset "same optimum from different starts" begin
        res2 = redirect_stdout(devnull) do
            Random.seed!(1); estimate_dynamic_model(dsmall)            # random start
        end
        @test Optim.converged(res2)
        @test res2.minimizer ≈ res.minimizer atol = 1e-3
        res3 = redirect_stdout(devnull) do
            estimate_dynamic_model(dsmall, θ_start = res.minimizer)   # start at the optimum
        end
        @test res3.minimizer ≈ res.minimizer atol = 1e-4
        @test Optim.iterations(res3) <= Optim.iterations(res)
    end
    @testset "β = 0 reproduces GLM on the same observations" begin
        d0  = merge(dsmall, (β = 0.0,))
        dfl = DataFrame(Y = vec(dsmall.Y), Odometer = vec(dsmall.X), Branded = repeat(dsmall.B, dsmall.T))
        g   = glm(@formula(Y ~ Odometer + Branded), dfl, Binomial(), LogitLink())
        r0  = redirect_stdout(devnull) do
            estimate_dynamic_model(d0, θ_start = θtest)
        end
        @test r0.minimizer ≈ coef(g) atol = 1e-3
        @test r0.minimum ≈ deviance(g) / 2 atol = 1e-4
    end
    @testset "Monte Carlo: recovers known parameters from simulated data" begin
        θtrue = [2.0, -0.15, 1.0]
        dsim  = simulate_panel(θtrue, d; N = 500, T = 10)
        @test 0.5 < mean(dsim.Y) < 0.98             # some replacements, mostly kept
        @test all(dsim.xval[dsim.Xstate] .≈ dsim.X)
        rsim = redirect_stdout(devnull) do
            estimate_dynamic_model(dsim, θ_start = [1.0, -0.05, 0.5])
        end
        @test Optim.converged(rsim)
        # MLE property: fitted likelihood is at least as good as at the truth
        @test rsim.minimum <= log_likelihood_dynamic(θtrue, dsim) + 1e-8
        # estimates are close to the truth (tolerances ≈ 3 standard errors at N·T = 5000)
        @test abs(rsim.minimizer[1] - θtrue[1]) < 0.25
        @test abs(rsim.minimizer[2] - θtrue[2]) < 0.03
        @test abs(rsim.minimizer[3] - θtrue[3]) < 0.30
        @test sign.(rsim.minimizer) == sign.(θtrue)
    end
end

#---------------------------------------------------
# End-to-end consistency of reported results
#---------------------------------------------------
@testset "end-to-end consistency" begin
    # static start from the GLM fit is what main() passes to the dynamic estimator
    @test coef(estimate_static_model(df_long)) ≈ θstat atol = 1e-2
    # reported dynamic optimum is a stationary point of the full-data likelihood
    h = 1e-4
    f(θ) = log_likelihood_dynamic(θ, d)
    grad = [(f(θhat .+ h .* (1:3 .== k)) - f(θhat .- h .* (1:3 .== k))) / (2h) for k in 1:3]
    @test maximum(abs.(grad)) < 1.0        # θhat is rounded to 4 decimals, so not exactly zero
    # dynamic coefficients are larger in magnitude than static ones, as discussed in the writeup
    @test all(abs.(θhat) .> abs.(θstat))
end

end # outer testset
