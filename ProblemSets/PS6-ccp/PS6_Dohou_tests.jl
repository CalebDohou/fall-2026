#---------------------------------------------------
# ECON 6343: Econometrics III — Problem Set 6 (Rust 1987, CCP estimation)
# Caleb Dohou
#
# Question 4: unit tests for every function in PS6_Dohou_source.jl.
# The source code is read in first, then the tests run.
#
# Run with:  julia PS6_Dohou_tests.jl
#
# Test design
#   * Each function gets its own @testset, in the order of the problem set.
#   * The data and the flexible logit are computed once and reused.
#   * Where possible a result is checked against an independent computation
#     or a property that must hold whatever the data: rows of xtran sum to 1
#     (so a constant FV gives FVT1 = 0), an offset equal to a*Odometer shifts
#     the Odometer coefficient by exactly -a, our own logit must equal GLM, etc.
#   * Small toy grids are used where a hand-made answer is possible.
#---------------------------------------------------

using Test
using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV

cd(@__DIR__)
include("create_grids.jl")
include("PS6_Dohou_source.jl")

################################################################################
# Fixtures and helpers
################################################################################

const URL = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS5-ddc/busdata.csv"
const β   = 0.9

println("Loading data and fitting the flexible logit (slow the first time: compilation)...")
const df_long, Xstate, Zstate, Branded = load_and_reshape_data(URL)
const N, T = size(Xstate)
const zval, zbin, xval, xbin, xtran = create_grids()
const flex_logit = estimate_flexible_logit(df_long)

logistic(η) = 1 ./ (1 .+ exp.(-η))

# Simulated logit data with an offset and known θ
function simulate_logit(n, θ; seed = 1234)
    rng    = MersenneTwister(seed)
    odo    = 10 .* rand(rng, n)
    brand  = Float64.(rand(rng, n) .< 0.5)
    offset = randn(rng, n)
    X      = hcat(ones(n), odo, brand)
    y      = Float64.(rand(rng, n) .< logistic(X*θ .+ offset))
    return X, y, offset
end

################################################################################
# Tests
################################################################################

@testset "PS6 Dohou: all tests" begin

#-------------------------------------------------------------------------------
@testset "load_and_reshape_data (Q1)" begin
    @testset "shape and columns" begin
        @test nrow(df_long) == 20_000
        @test (N, T) == (1000, 20)
        for c in [:bus_id, :time, :Y, :Odometer, :RouteUsage, :Branded]
            @test c in propertynames(df_long)
        end
        @test !any(ismissing, Matrix(df_long))
    end

    @testset "sorted by bus then time (needed for FVT1'[:])" begin
        @test df_long.bus_id == repeat(1:N, inner = T)
        @test df_long.time   == repeat(1.0:T, N)
    end

    @testset "values" begin
        @test all(in((0, 1)), df_long.Y)
        @test all(in((0, 1)), df_long.Branded)
        @test all(df_long.Odometer .>= 0)
        @test all(in((0, 1)), Branded)
    end

    @testset "bus-level variables constant within bus" begin
        g = combine(groupby(df_long, :bus_id),
                    :RouteUsage => (x -> length(unique(x))) => :nru,
                    :Branded    => (x -> length(unique(x))) => :nb)
        @test all(g.nru .== 1)
        @test all(g.nb  .== 1)
    end

    @testset "wide states are consistent with long data and the grids" begin
        @test eltype(Xstate) <: Integer
        @test eltype(Zstate) <: Integer
        @test all(1 .<= Xstate .<= xbin)
        @test all(1 .<= Zstate .<= zbin)
        odo_wide = permutedims(reshape(df_long.Odometer, T, N))
        @test xval[Xstate] ≈ odo_wide
        ru = df_long.RouteUsage[1:T:end]
        @test zval[Zstate] ≈ ru
        @test Branded == df_long.Branded[1:T:end]
    end
end

#-------------------------------------------------------------------------------
@testset "estimate_flexible_logit (Q2)" begin
    b = coef(flex_logit)

    @testset "128 terms, some dropped by GLM" begin
        @test length(b) == 128
        ndrop = sum(b .== 0)
        @test 0 < ndrop < 128
        @test all(isfinite, b)
    end

    @testset "the drop is numerical, not true collinearity" begin
        # With the PDF formula GLM drops the intercept itself ...
        @test coefnames(flex_logit)[1] == "(Intercept)"
        @test b[1] == 0
        # ... but once each column is scaled to unit length the model matrix
        # has full rank 128: the dropped columns are not truly collinear.
        X  = modelmatrix(flex_logit)
        Xs = X ./ sqrt.(sum(X.^2, dims = 1))
        @test rank(Xs) == 128
        # Without an intercept, the fitted mean no longer equals mean(Y)
        # exactly (it would for a logit with an intercept). It is still close.
        p = predict(flex_logit)
        @test mean(p) ≈ mean(df_long.Y) atol = 0.01
    end

    @testset "fitted probabilities" begin
        p = predict(flex_logit)
        @test all(0 .<= p .<= 1)
    end

    @testset "score equations X'(y - p) = 0 hold" begin
        X = modelmatrix(flex_logit)
        p = predict(flex_logit)
        keep = b .!= 0
        score = X[:, keep]' * (df_long.Y .- p)
        @test maximum(abs.(score ./ (abs.(X[:, keep])' * ones(nrow(df_long))))) < 1e-5
    end

    @testset "predict() recomputes squared terms on new data" begin
        newdf = copy(df_long)
        newdf.time .= 7.0
        newdf.Odometer .= 3.0
        Xnew = GLM.StatsModels.modelcols(formula(flex_logit).rhs, newdf)
        @test predict(flex_logit, newdf) ≈ logistic(Xnew * b)
        # the Odometer^2 column must be 9 everywhere, time^2 must be 49
        names = coefnames(flex_logit)
        @test all(Xnew[:, findfirst(==("Odometer ^ 2"), names)] .== 9.0)
        @test all(Xnew[:, findfirst(==("time ^ 2"), names)] .== 49.0)
    end

    @testset "economic content: more mileage, more replacement" begin
        lo = copy(df_long); lo.Odometer .= 2.0
        hi = copy(df_long); hi.Odometer .= 15.0
        @test mean(predict(flex_logit, hi)) < mean(predict(flex_logit, lo))
    end
end

#-------------------------------------------------------------------------------
@testset "create_grids (Q3a)" begin
    @test size(xtran) == (xbin*zbin, xbin)
    @test all(xtran .>= 0)
    @test all(isapprox.(sum(xtran, dims = 2), 1; atol = 1e-10))
    # mileage never goes down: entries below the diagonal of each block are 0
    for z in (1, zbin), x in 2:xbin
        @test all(xtran[(z-1)*xbin + x, 1:x-1] .== 0)
    end
end

#-------------------------------------------------------------------------------
@testset "construct_state_space (Q3b)" begin
    @testset "real grid: same rows as xtran" begin
        s = construct_state_space(xbin, zbin, xval, zval)
        @test nrow(s) == size(xtran, 1)
        @test propertynames(s) == [:Odometer, :RouteUsage, :Branded, :time]
        @test all(s.Branded .== 0)
        @test all(s.time .== 0)
        for z in (1, 17, zbin), x in (1, 50, xbin)
            row = (z-1)*xbin + x
            @test s.Odometer[row]   == xval[x]
            @test s.RouteUsage[row] == zval[z]
        end
    end

    @testset "toy grid by hand" begin
        s = construct_state_space(2, 3, [0.0, 1.0], [10.0, 20.0, 30.0])
        @test s.Odometer   == [0, 1, 0, 1, 0, 1]
        @test s.RouteUsage == [10, 10, 20, 20, 30, 30]
    end
end

#-------------------------------------------------------------------------------
@testset "compute_future_values (Q3b)" begin
    s  = construct_state_space(xbin, zbin, xval, zval)
    FV = compute_future_values(s, flex_logit, xtran, T, β)

    @testset "shape, unused slices, finiteness, sign" begin
        @test size(FV) == (size(xtran, 1), 2, T + 1)
        @test all(FV[:, :, 1]   .== 0)
        @test all(FV[:, :, T+1] .== 0)
        @test all(isfinite, FV)
        @test all(FV .>= 0)                 # -β log p0 ≥ 0 since p0 ≤ 1
        @test maximum(FV) <= -β*log(eps()) + 1e-12   # bound from eps()
    end

    @testset "matches an independent recomputation" begin
        for (t, b) in ((2, 0), (11, 1), (T, 0))
            s2 = construct_state_space(xbin, zbin, xval, zval)
            s2.time .= t; s2.Branded .= b
            p0 = 1 .- predict(flex_logit, s2)
            @test FV[:, b+1, t] ≈ -β .* log.(max.(p0, eps()))
        end
    end

    @testset "β scaling and β = 0" begin
        s2 = construct_state_space(xbin, zbin, xval, zval)
        @test all(compute_future_values(s2, flex_logit, xtran, 3, 0.0) .== 0)
        FV45 = compute_future_values(s2, flex_logit, xtran, 3, 0.45)
        FV90 = compute_future_values(s2, flex_logit, xtran, 3, 0.9)
        @test FV90 ≈ 2 .* FV45
        @test FV90[:, :, 2:3] ≈ FV[:, :, 2:3]   # same as long run for t ≤ 3
    end
end

#-------------------------------------------------------------------------------
@testset "compute_fvt1 (Q3b)" begin
    @testset "constant FV gives zero (rows of xtran sum to 1)" begin
        FVc  = fill(3.7, size(xtran, 1), 2, T + 1)
        fvt1 = compute_fvt1(df_long, FVc, xtran, Xstate, Zstate, xbin, Branded)
        @test maximum(abs.(fvt1)) < 1e-10
    end

    @testset "mileage state 1 (= replacement state) gives zero" begin
        X1   = ones(Int, N, T)
        FVr  = rand(MersenneTwister(1), size(xtran, 1), 2, T + 1)
        fvt1 = compute_fvt1(df_long, FVr, xtran, X1, Zstate, xbin, Branded)
        @test all(fvt1 .== 0)
    end

    @testset "toy example by hand, ordering bus-then-time" begin
        # 2 mileage bins, 2 route usage bins, 2 buses, 2 periods
        xt = [0.5 0.5;    # z=1, x=1
              0.0 1.0;    # z=1, x=2
              0.2 0.8;    # z=2, x=1
              0.0 1.0]    # z=2, x=2
        FVt = zeros(4, 2, 3)
        FVt[:, 1, 2] = [1.0, 2.0, 3.0, 4.0];  FVt[:, 2, 2] = [10.0, 20.0, 30.0, 40.0]
        FVt[:, 1, 3] = [5.0, 7.0, 6.0, 9.0];  FVt[:, 2, 3] = [50.0, 70.0, 60.0, 90.0]
        Xs  = [2 1; 2 2]        # bus 1: x=2 then x=1; bus 2: x=2 both
        Zs  = [1, 2]
        Bs  = [0, 1]
        dfl = DataFrame(bus_id = [1, 1, 2, 2], time = [1, 2, 1, 2])
        out = compute_fvt1(dfl, FVt, xt, Xs, Zs, 2, Bs)
        # bus 1 (z=1, b=0), t=1, x=2: (xt[2,:]-xt[1,:])'*FV[1:2,1,2] = (-0.5,0.5)'(1,2)
        # bus 1, t=2, x=1: 0
        # bus 2 (z=2, b=1), t=1, x=2: (xt[4,:]-xt[3,:])'*FV[3:4,2,2] = (-0.2,0.2)'(30,40)
        # bus 2, t=2, x=2: (-0.2,0.2)'(60,90)
        @test out ≈ [0.5, 0.0, 2.0, 6.0]
    end

    @testset "real data: length and finiteness" begin
        s    = construct_state_space(xbin, zbin, xval, zval)
        FV   = compute_future_values(s, flex_logit, xtran, T, β)
        fvt1 = compute_fvt1(df_long, FV, xtran, Xstate, Zstate, xbin, Branded)
        @test length(fvt1) == nrow(df_long)
        @test all(isfinite, fvt1)
        # last period has no future
        @test all(fvt1[T:T:end] .== 0)
    end
end

#-------------------------------------------------------------------------------
@testset "estimate_structural_params (Q3c)" begin
    @testset "zero offset equals a plain logit" begin
        m0 = estimate_structural_params(df_long, zeros(nrow(df_long)))
        m1 = glm(@formula(Y ~ Odometer + Branded), df_long, Binomial(), LogitLink())
        @test coef(m0) ≈ coef(m1)
        @test coefnames(m0) == ["(Intercept)", "Odometer", "Branded"]
    end

    @testset "offset has coefficient exactly 1" begin
        # offset = a*Odometer moves the Odometer coefficient by exactly -a
        a  = 0.05
        m0 = estimate_structural_params(df_long, zeros(nrow(df_long)))
        ma = estimate_structural_params(df_long, a .* df_long.Odometer)
        @test coef(ma)[2] ≈ coef(m0)[2] - a  atol = 1e-6
        @test coef(ma)[[1, 3]] ≈ coef(m0)[[1, 3]]  atol = 1e-6
    end

    @testset "does not modify df_long" begin
        @test !(:fv in propertynames(df_long))
    end
end

#-------------------------------------------------------------------------------
@testset "logit_offset_loglike (Q3d)" begin
    X, y, off = simulate_logit(500, [0.5, -0.2, 1.0])
    θ = [0.3, -0.1, 0.8]

    @test logit_offset_loglike(θ, X, y, off) ≈
          sum(y .* log.(logistic(X*θ .+ off)) .+ (1 .- y) .* log.(1 .- logistic(X*θ .+ off)))
    @test logit_offset_loglike(zeros(3), X, y, zeros(500)) ≈ -500*log(2)
    @test logit_offset_loglike(θ, X, y, off) < 0
    # stable for very large indices (naive formula gives -Inf or NaN)
    @test isfinite(logit_offset_loglike([800.0, 0, 0], X, y, off))
    @test isfinite(logit_offset_loglike([-800.0, 0, 0], X, y, off))
end

#-------------------------------------------------------------------------------
@testset "estimate_logit_offset (Q3d)" begin
    θtrue = [0.5, -0.2, 1.0]
    X, y, off = simulate_logit(20_000, θtrue)
    θ̂, se, ll = estimate_logit_offset(X, y, off)

    @testset "matches GLM with offset" begin
        d = DataFrame(y = y, Odometer = X[:, 2], Branded = X[:, 3])
        m = glm(@formula(y ~ Odometer + Branded), d, Binomial(), LogitLink(), offset = off)
        @test θ̂ ≈ coef(m)    rtol = 1e-5
        @test se ≈ stderror(m) rtol = 1e-3
        @test ll ≈ loglikelihood(m) rtol = 1e-8
    end

    @testset "recovers the true θ" begin
        @test all(abs.(θ̂ .- θtrue) .< 4 .* se)
    end

    @testset "first-order conditions" begin
        @test maximum(abs.(X' * (y .- logistic(X*θ̂ .+ off)))) < 1e-3
    end

    @testset "other starting value gives the same answer" begin
        θ̂2, _, _ = estimate_logit_offset(X, y, off; θ0 = [2.0, 0.5, -1.0])
        @test θ̂2 ≈ θ̂ rtol = 1e-5
    end
end

#-------------------------------------------------------------------------------
@testset "main (Q3e, end to end)" begin
    m, θ̂_own = redirect_stdout(devnull) do
        main()
    end
    @test coef(m) ≈ [1.944, -0.1487, 1.1249] atol = 1e-3
    @test θ̂_own ≈ coef(m) rtol = 1e-5
    @test coef(m)[2] < 0      # more mileage, less utility of running
    @test coef(m)[3] > 0      # branded buses are kept more
end

end # all tests
