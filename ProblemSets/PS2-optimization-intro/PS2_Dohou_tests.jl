#*********************************************************
# ECON 6343: Econometrics III
# Problem Set 2: question 7 -- unit tests
# Author: Caleb Dohou
#
# This file loads the packages, includes the source file and tests it.
# It downloads the data and runs the question 5 estimation, so it takes a
# few minutes.
#*********************************************************

using Test
using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables

include("PS2_Dohou_source.jl")

# The functions print a lot. capture() runs a function and saves what it
# prints in a temporary file. Then the test output is clean, and I can
# check the printed text.
function capture(f)
    logfile = tempname()
    ret = open(logfile, "w") do io
        redirect_stdout(io) do
            f()
        end
    end
    return ret, read(logfile, String)
end

# load the data one time for all the tests
df = load_nlsw88()
X = build_X(df)
y = df.married .== 1
N = size(X, 1)

# naive logit negative log-likelihood with p_i, like in the Lecture 4
# slides. I use it to test logit_like().
function logit_like_naive(beta, X, y)
    p = exp.(X*beta) ./ (1 .+ exp.(X*beta))
    return -sum(y .* log.(p) .+ (1 .- y) .* log.(1 .- p))
end


#***************************************************
# tests for the data helpers
#***************************************************
@testset "load_nlsw88() and build_X()" begin

    #--------------------------
    # the data have the right number of rows, and white is correct
    #--------------------------
    @test nrow(df) == 2246
    @test "white" in names(df)
    @test df.white == (df.race .== 1)

    #--------------------------
    # build_X: N x 4, Float64, and the columns are in the right order
    #--------------------------
    @test size(X) == (2246, 4)
    @test eltype(X) == Float64
    @test all(X[:,1] .== 1.0)
    @test X[:,2] == df.age
    @test X[:,3] == (df.race .== 1)
    @test X[:,4] == (df.collgrad .== 1)
end

@testset "printpretty()" begin
    ret, printed = capture(() -> printpretty([1.0 2.0; 3.0 4.0]))
    @test ret === nothing
    # it prints like the REPL, not in one line like println
    @test occursin("2×2 Matrix{Float64}", printed)
    @test !occursin("[1.0 2.0; 3.0 4.0]", printed)
end


#***************************************************
# tests for question 1
#***************************************************
@testset "question 1: f(), negf(), q1()" begin

    #--------------------------
    # f is the polynomial of question 1, and negf is its negative
    #--------------------------
    @test f([0.0]) == -2.0
    @test f([1.0]) == -1 - 10 - 2 - 3 - 2
    @test f([-1.0]) == -1 + 10 - 2 + 3 - 2
    for x in (-10.0, -7.3, -0.5, 0.0, 0.5, 3.0)
        @test negf([x]) == -f([x])
    end

    #--------------------------
    # q1 finds the maximum from a normal starting value
    #--------------------------
    (xstar, fstar), printed = capture(() -> q1(startval = [0.5]))
    @test xstar ≈ -7.378243405 atol=1e-6
    @test fstar ≈ 964.3133838 atol=1e-4
    @test fstar ≈ f([xstar])
    # the first-order condition f'(x*) = 0
    @test -(4xstar^3 + 30xstar^2 + 4xstar + 3) ≈ 0.0 atol=1e-4
    @test occursin("argmax of f(x) is", printed)

    #--------------------------
    # it is the global maximum: no point on a fine grid is higher
    #--------------------------
    @test maximum(f([x]) for x in -20:0.001:20) <= fstar + 1e-8

    #--------------------------
    # it also works from bad starting values, because negf' changes sign
    # only one time
    #--------------------------
    for s in (-1000.0, -20.0, 100.0, 1e5)
        (xs, _), _ = capture(() -> q1(startval = [s]))
        @test xs ≈ -7.378243405 atol=1e-5
    end
end


#***************************************************
# tests for question 2
#***************************************************
@testset "question 2: ols(), q2()" begin

    #--------------------------
    # ols returns one number, the sum of squared residuals
    #--------------------------
    @test ols(zeros(4), X, y) isa Real
    # when beta = 0, each residual is y_i, which is 0 or 1
    @test ols(zeros(4), X, y) == sum(y)
    Random.seed!(11)
    b = randn(4) ./ 10
    @test ols(b, X, y) ≈ sum((y .- X*b).^2)

    #--------------------------
    # a perfect fit gives SSR = 0, other values give a bigger SSR
    #--------------------------
    Xs = [ones(5) collect(1.0:5.0)]
    ys = Xs*[2.0, 3.0]
    @test ols([2.0, 3.0], Xs, ys) ≈ 0.0 atol=1e-20
    @test ols([2.0, 3.5], Xs, ys) > 0.0

    #--------------------------
    # the three methods in q2 give the same result, and the standard errors
    # are the same as in lm()
    #--------------------------
    res, printed = capture(() -> q2(X, y, df))
    bc = inv(X'*X)*X'*y
    @test res.closed_form == bc
    @test res.optim ≈ bc atol=1e-6
    @test coef(res.lm) ≈ bc atol=1e-10
    @test res.se ≈ stderror(res.lm) rtol=1e-8
    @test res.closed_form ≈ [0.661351, -0.00462591, 0.225950, -0.0121842] atol=1e-6

    # the closed form is the minimum: a small change in any coefficient
    # makes SSR bigger
    for k in 1:4
        e = zeros(4); e[k] = 1e-3
        @test ols(bc .+ e, X, y) > ols(bc, X, y)
        @test ols(bc .- e, X, y) > ols(bc, X, y)
    end

    #--------------------------
    # the output prints the two differences used in the writeup
    #--------------------------
    @test occursin("largest gap between the two", printed)
    @test occursin("largest gap between lm() and the closed form", printed)
end


#***************************************************
# tests for question 3
#***************************************************
@testset "question 3: logit_like(), q3()" begin

    #--------------------------
    # when beta = 0, all p_i are 1/2, so the result is N*log(2)
    #--------------------------
    @test logit_like(zeros(4), X, y) ≈ N*log(2)

    #--------------------------
    # the stable version gives the same value as the p_i formula, when the
    # p_i formula works
    #--------------------------
    @test logit_like([0.5, -0.01, 0.8, -0.05], X, y) ≈
          logit_like_naive([0.5, -0.01, 0.8, -0.05], X, y)
    Random.seed!(22)
    for _ in 1:5
        b = randn(4) ./ 20
        @test logit_like(b, X, y) ≈ logit_like_naive(b, X, y)
    end

    #--------------------------
    # why I need the stable version: at b = ones(4), exp(X*b) is finite but
    # p_i becomes exactly 1.0, so log(1 - p_i) is -Inf and the naive version
    # gives NaN. The stable version is finite, also for very large values.
    #--------------------------
    Xb = X*ones(4)
    p = exp.(Xb) ./ (1 .+ exp.(Xb))
    @test !any(isinf.(exp.(Xb)))
    @test any(p .== 1.0)
    @test isnan(logit_like_naive(ones(4), X, y))
    @test isfinite(logit_like(ones(4), X, y))
    @test isfinite(logit_like(fill(1e6, 4), X, y))
    @test isfinite(logit_like(fill(-1e6, 4), X, y))

    #--------------------------
    # at the glm coefficients, logit_like gives the same log-likelihood as glm
    #--------------------------
    mglm = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
    @test -logit_like(coef(mglm), X, y) ≈ loglikelihood(mglm) rtol=1e-10

    #--------------------------
    # q3 gives the glm estimates, from a normal start and from bad starts
    #--------------------------
    b3, printed = capture(() -> q3(X, y))
    @test b3 ≈ coef(mglm) atol=1e-5
    @test occursin("logit estimates from Optim", printed)
    for s in (zeros(4), fill(1.0, 4), fill(-1.0, 4))
        bs, _ = capture(() -> q3(X, y; startval = s))
        @test bs ≈ coef(mglm) atol=1e-4
    end
end


#***************************************************
# tests for question 4
#***************************************************
@testset "question 4: q4()" begin
    b3, _ = capture(() -> q3(X, y; startval = zeros(4)))
    res, printed = capture(() -> q4(df, b3))

    # the glm coefficients are in the same order as the columns of X,
    # so q4 can compare them one by one
    @test coefnames(res.glm) == ["(Intercept)", "age", "white", "collgrad"]

    # q4 computes the difference correctly, and it is small
    @test res.maxgap == maximum(abs.(coef(res.glm) .- b3))
    @test res.maxgap < 1e-5
    @test occursin("largest gap between the glm and Optim estimates", printed)
end


#***************************************************
# tests for question 5
#***************************************************
dfm, _ = capture(() -> clean_occupation(df))
Xm = build_X(dfm)
ym = dfm.occupation
Nm = size(Xm, 1)

@testset "question 5: clean_occupation()" begin
    ret, printed = capture(() -> clean_occupation(df))

    #--------------------------
    # missing occupations are dropped, and 8 to 13 are now 7
    #--------------------------
    @test nrow(ret) == 2237
    @test !any(ismissing, ret.occupation)
    @test sort(unique(ret.occupation)) == 1:7
    @test [count(==(j), ret.occupation) for j in 1:7] == [317, 264, 726, 102, 53, 246, 529]

    #--------------------------
    # the original DataFrame df does not change
    #--------------------------
    @test count(isequal(8), df.occupation) == 286
    @test count(isequal(13), df.occupation) == 187
    @test count(ismissing, df.occupation) == 9

    #--------------------------
    # the frequency tables print with labels, also the missing row
    #--------------------------
    @test occursin("Named Vector", printed)
    @test occursin("missing", printed)
end

@testset "question 5: mlogit_probs(), mlogit_like(), mlogit_like_loop()" begin

    #--------------------------
    # when alpha = 0, every alternative has probability 1/7
    #--------------------------
    @test mlogit_like(zeros(24), Xm, ym) ≈ Nm*log(7)
    @test all(mlogit_probs(zeros(24), Xm, 7) .≈ 1/7)

    #--------------------------
    # the probabilities are between 0 and 1 and sum to 1, and the
    # likelihood uses them
    #--------------------------
    Random.seed!(33)
    a = randn(24) ./ 20
    P = mlogit_probs(a, Xm, 7)
    @test size(P) == (Nm, 7)
    @test all(0 .< P .< 1)
    @test all(sum(P, dims=2) .≈ 1.0)
    @test mlogit_like(a, Xm, ym) ≈ -sum(log(P[i, ym[i]]) for i in 1:Nm)

    #--------------------------
    # mlogit_like gives the same value as the loop version
    #--------------------------
    for _ in 1:3
        a = randn(24) ./ 20
        @test mlogit_like(a, Xm, ym) ≈ mlogit_like_loop(a, Xm, ym) rtol=1e-10
    end

    #--------------------------
    # with two alternatives, the multinomial logit is the same as the logit
    #--------------------------
    y2 = [v == 1 ? 1 : 2 for v in ym]
    for _ in 1:3
        b = randn(4) ./ 20
        @test mlogit_like(b, Xm, y2) ≈ logit_like(b, Xm, y2 .== 1)
    end

    #--------------------------
    # check the reshape used in the question 5 table: rows are the
    # covariates, columns are the alternatives
    #--------------------------
    a = zeros(24)
    a[(3-1)*4 + 1] = 5.0            # intercept (row 1) of alternative 3
    P = mlogit_probs(a, Xm, 7)
    @test all(argmax(P[i, :]) == 3 for i in 1:Nm)
    a = zeros(24)
    a[(5-1)*4 + 2] = 1.0            # age (row 2) of alternative 5
    P = mlogit_probs(a, Xm, 7)
    @test all(argmax(P[i, :]) == 5 for i in 1:Nm)

    #--------------------------
    # no overflow for very large parameter values
    #--------------------------
    @test isfinite(mlogit_like(fill(1e6, 24), Xm, ym))
    @test isfinite(mlogit_like(fill(-1e6, 24), Xm, ym))
end

@testset "question 5: q5(), q5_checks()" begin

    #--------------------------
    # q5 returns the estimates with full precision, as a 4 x 6 matrix
    #--------------------------
    mnl, printed = capture(() -> q5(Xm, ym))
    @test size(mnl.alpha) == (4, 6)
    @test mnl.loglike ≈ -mlogit_like(vec(mnl.alpha), Xm, ym)
    @test mnl.loglike ≈ -3605.0904 atol=1e-3

    # the estimates in the writeup, with 4 decimals
    reported = [ 0.1910 -0.1699  0.6895 -2.2675 -1.3986  0.2455;
                -0.0335 -0.0360 -0.0105 -0.0053 -0.0143 -0.0067;
                 0.5964  1.3068  0.5232  1.3914 -0.0177 -0.5383;
                 0.4165 -0.4310 -1.4925 -0.9850 -1.4951 -3.7898]
    @test mnl.alpha ≈ reported atol=5e-4

    # it prints the rounded estimates and the stopping criteria
    @test occursin("rounded to 4 decimals", printed)
    @test occursin("stopping criteria met", printed)

    #--------------------------
    # q5_checks passes at the estimates
    #--------------------------
    checks, _ = capture(() -> q5_checks(Xm, ym, mnl.alpha))
    @test checks.maxscore < 1e-3
    @test checks.maxsharegap < 1e-6
    @test checks.loopgap < 1e-8

    #--------------------------
    # and it fails when the estimates are wrong
    #--------------------------
    bad, _ = capture(() -> q5_checks(Xm, ym, mnl.alpha .+ 0.1))
    @test bad.maxscore > 1.0
    @test bad.maxsharegap > 1e-3
end


#***************************************************
# tests for question 6
#***************************************************
@testset "question 6: allwrap(), including q5_startvals()" begin
    ret, printed = capture(allwrap)
    @test ret === nothing

    # all questions run in order, and allwrap prints that it finished
    headers = ["question 1:", "question 2:", "question 3:", "question 4:",
               "question 5:", "question 5 check:", "question 5 starting values:",
               "allwrap() finished"]
    positions = [findfirst(h, printed) for h in headers]
    @test all(!isnothing, positions)
    @test issorted(first.(positions))

    #--------------------------
    # q5_startvals: the U[0,1] and U[-1,1] starts give the same maximum as
    # the zero start, up to 4 decimals
    #--------------------------
    for name in ("U\\[0,1\\]", "U\\[-1,1\\]")
        m = match(Regex("from a $name vector, log-likelihood (\\S+), largest gap from the zero-start estimates (\\S+)"),
                  printed)
        @test m !== nothing
        @test parse(Float64, m.captures[1]) ≈ -3605.0904 atol=1e-3
        @test parse(Float64, m.captures[2]) < 5e-4
    end
end
