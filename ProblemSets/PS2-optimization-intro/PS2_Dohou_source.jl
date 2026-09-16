#*************************************************
# ECON 6343: Econometrics III
# Problem Set 2: source code (functions only)
# Author: Caleb Dohou
#*************************************************

# This file has only functions. The packages are loaded in
# PS2_Dohou_script.jl and PS2_Dohou_tests.jl, not here.

#***************************************************
# data
#***************************************************
"""
    NLSW88_URL

URL of the nlsw88.csv data used in questions 2 to 5.

"""
const NLSW88_URL = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS1-julia-intro/nlsw88.csv"

"""
    load_nlsw88(url=NLSW88_URL)

Reads nlsw88.csv from the URL and adds the `white` dummy (race == 1).

Returns the DataFrame.

"""
function load_nlsw88(url::AbstractString = NLSW88_URL)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    df.white = df.race .== 1
    return df
end

"""
    printpretty(x)

Prints `x` like the REPL does, so tables and matrices are easy to read.

Returns nothing.

"""
function printpretty(x)
    show(stdout, MIME("text/plain"), x)
    println()
    return nothing
end

"""
    build_X(df)

Builds the N x 4 matrix of covariates: intercept, age, white, collgrad.

"""
function build_X(df)
    return [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
end


#***************************************************
# question 1
#***************************************************
"""
    f(x)

The function of question 1, f(x) = -x^4 - 10x^3 - 2x^2 - 3x - 2. `x` is a
one-element vector.

"""
f(x) = -x[1]^4 - 10x[1]^3 - 2x[1]^2 - 3x[1] - 2

"""
    negf(x)

The negative of `f`. Optim only minimizes, so I minimize `negf` to maximize `f`.

"""
negf(x) = x[1]^4 + 10x[1]^3 + 2x[1]^2 + 3x[1] + 2

"""
    q1(; startval=rand(1))

Answers question 1. Maximizes `f` by minimizing `negf` with LBFGS().

Returns `(xstar, fstar)`: the argmax and the maximum.

"""
function q1(; startval = rand(1))
    result = optimize(negf, startval, LBFGS())

    # Optim minimized -f, so the maximum of f is minus the minimum
    xstar = Optim.minimizer(result)[1]
    fstar = -Optim.minimum(result)

    println("question 1: optimization summary:")
    println(result)
    println("question 1: argmax of f(x) is ", xstar)
    println("question 1: max of f(x) is ", fstar)

    return xstar, fstar
end


#***************************************************
# question 2
#***************************************************
"""
    ols(beta, X, y)

Sum of squared residuals at `beta`. This is the OLS objective.

"""
function ols(beta, X, y)
    ssr = (y .- X*beta)'*(y .- X*beta)
    return ssr
end

"""
    q2(X, y, df; startval=rand(size(X,2)))

Answers question 2. Estimates the linear probability model of `married` on
age, white and collgrad in three ways: Optim with `ols`, the closed form
inv(X'X)X'y, and lm(). Also computes the OLS standard errors.

Returns `(optim, closed_form, lm, se)`.

"""
function q2(X, y, df; startval = rand(size(X,2)))
    # use a closure to give the data X and y to Optim
    beta_hat_ols = optimize(b -> ols(b, X, y), startval, LBFGS(),
                            Optim.Options(g_tol=1e-6, iterations=100_000,
                                          show_trace=false))
    b_optim = Optim.minimizer(beta_hat_ols)

    # closed-form solution, Optim should give the same result
    b_closed = inv(X'*X)*X'*y

    # the same regression with lm() from GLM
    b_lm = lm(@formula(married ~ age + white + collgrad), df)

    # OLS standard errors
    σ² = sum((y .- X*b_closed).^2)/(size(X,1) - size(X,2))
    vcov_b = σ²*inv(X'*X)
    se = sqrt.(diag(vcov_b))

    println("question 2: OLS estimates from Optim:")
    println(b_optim)
    println("question 2: OLS estimates in closed form, inv(X'X)X'y:")
    println(b_closed)
    println("question 2: largest gap between the two: ",
            maximum(abs.(b_optim .- b_closed)))
    println("question 2: OLS estimates from lm():")
    printpretty(coeftable(b_lm))
    println("question 2: largest gap between lm() and the closed form: ",
            maximum(abs.(coef(b_lm) .- b_closed)))
    println("question 2: [estimate standard_error] by hand:")
    printpretty([b_closed se])

    return (optim = b_optim, closed_form = b_closed, lm = b_lm, se = se)
end


#***************************************************
# question 3
#***************************************************
"""
    logit_like(beta, X, y)

Negative log-likelihood of the binary logit at `beta`. It is negative because
Optim minimizes.

It uses log L = sum(y*Xb - log(1 + exp(Xb))). The term log(1 + exp(Xb)) is
computed as max(Xb, 0) + log(1 + exp(-|Xb|)), so it does not overflow.

"""
function logit_like(beta, X, y)
    Xb = X*beta

    # log(1 + exp(Xb)) in a stable form, so exp() does not overflow
    logdenom = max.(Xb, 0) .+ log1p.(exp.(-abs.(Xb)))

    loglike = sum(y .* Xb .- logdenom)

    return -loglike
end

"""
    q3(X, y; startval=rand(size(X,2)))

Answers question 3. Estimates the logit of `married` with Optim, using
`logit_like`.

Returns the estimated coefficients.

"""
function q3(X, y; startval = rand(size(X,2)))
    beta_hat_logit = optimize(b -> logit_like(b, X, y), startval, LBFGS(),
                              Optim.Options(g_tol=1e-6, iterations=100_000,
                                            show_trace=false))
    b_logit = Optim.minimizer(beta_hat_logit)

    println("question 3: logit estimates from Optim:")
    println(b_logit)
    println("question 3: log-likelihood at the optimum: ",
            -Optim.minimum(beta_hat_logit))

    return b_logit
end


#***************************************************
# question 4
#***************************************************
"""
    q4(df, b_logit)

Answers question 4. Estimates the same logit with glm() and compares it to
`b_logit`, the Optim estimates from `q3`.

Returns `(glm, maxgap)`: the glm model and the largest difference between the
two.

"""
function q4(df, b_logit)
    blogit_glm = glm(@formula(married ~ age + white + collgrad), df,
                     Binomial(), LogitLink())

    # compare glm and Optim, the coefficients should be the same
    maxgap = maximum(abs.(coef(blogit_glm) .- b_logit))

    println("question 4: logit estimates from glm():")
    printpretty(coeftable(blogit_glm))
    println("question 4: largest gap between the glm and Optim estimates: ",
            maxgap)

    return (glm = blogit_glm, maxgap = maxgap)
end


#***************************************************
# question 5
#***************************************************
"""
    clean_occupation(df)

Prepares the data for question 5. Drops rows with missing occupation and puts
occupations 8 to 13 into occupation 7. Prints the frequency table before and
after.

Returns a new DataFrame. `df` is not changed.

"""
function clean_occupation(df)
    println("question 5: occupation before aggregating:")
    # some occupations have very few observations
    printpretty(freqtable(df, :occupation))

    dfm = dropmissing(df, :occupation)
    dfm[dfm.occupation.==8 ,:occupation] .= 7
    dfm[dfm.occupation.==9 ,:occupation] .= 7
    dfm[dfm.occupation.==10,:occupation] .= 7
    dfm[dfm.occupation.==11,:occupation] .= 7
    dfm[dfm.occupation.==12,:occupation] .= 7
    dfm[dfm.occupation.==13,:occupation] .= 7

    println("question 5: occupation after aggregating:")
    # now all occupations have enough observations
    printpretty(freqtable(dfm, :occupation))

    return dfm
end

"""
    mlogit_like(alpha, X, y)

Negative log-likelihood of the multinomial logit at `alpha`. The last
alternative J is the base.

`alpha` has K(J-1) parameters. It is reshaped into a K x (J-1) matrix, and a
column of zeros is added for the base. Each row's maximum is subtracted before
`exp`, so it does not overflow.

"""
function mlogit_like(alpha, X, y)
    N = size(X, 1)          # number of observations
    K = size(X, 2)          # number of covariates
    J = maximum(y)          # number of alternatives, J is the base

    # make the K x J matrix of parameters, with zeros for the base
    alpha_mat = [reshape(alpha, K, J-1) zeros(eltype(alpha), K)]

    # N x J matrix of X*alpha, the last column is zero
    Xa = X*alpha_mat

    # log of the denominator (subtract the row maximum to avoid overflow)
    rowmax = maximum(Xa, dims=2)
    logdenom = rowmax .+ log.(sum(exp.(Xa .- rowmax), dims=2))

    # d[i,j] = 1 if person i chose alternative j
    d = [y[i] == j for i in 1:N, j in 1:J]

    loglike = sum(d .* (Xa .- logdenom))

    return -loglike
end

"""
    mlogit_probs(alpha, X, J)

The N x J matrix of multinomial logit probabilities at `alpha`, with the same
setup as `mlogit_like`. Each row sums to 1.

"""
function mlogit_probs(alpha, X, J)
    K = size(X, 2)
    alpha_mat = [reshape(alpha, K, J-1) zeros(eltype(alpha), K)]

    Xa = X*alpha_mat
    rowmax = maximum(Xa, dims=2)
    expXa = exp.(Xa .- rowmax)

    return expXa ./ sum(expXa, dims=2)
end

"""
    mlogit_like_loop(alpha, X, y)

A simple loop version of `mlogit_like`, written directly from the formula. I
use it only to check `mlogit_like`.

"""
function mlogit_like_loop(alpha, X, y)
    K = size(X, 2)
    J = maximum(y)
    alpha_mat = [reshape(alpha, K, J-1) zeros(K)]

    loglike = 0.0
    for i in axes(X, 1)
        denom = 0.0
        for j in 1:J
            denom += exp(sum(X[i, k]*alpha_mat[k, j] for k in 1:K))
        end
        numer = exp(sum(X[i, k]*alpha_mat[k, y[i]] for k in 1:K))
        loglike += log(numer/denom)
    end

    return -loglike
end

"""
    fit_mlogit(X, y, startval)

Minimizes `mlogit_like` with LBFGS() and g_tol = 1e-5, starting from
`startval`. `q5` and `q5_startvals` both use it, so they have the same
settings.

Returns the Optim result.

"""
function fit_mlogit(X, y, startval)
    return optimize(a -> mlogit_like(a, X, y), startval, LBFGS(),
                    Optim.Options(g_tol=1e-5, iterations=100_000,
                                  show_trace=false))
end

"""
    q5(X, y; startval=zeros(size(X,2)*(maximum(y)-1)))

Answers question 5. Estimates the multinomial logit of occupation, starting
from zeros. Prints the estimates rounded to 4 decimals and which stopping
criteria were met.

Returns `(alpha, loglike, result)`: the K x (J-1) matrix of estimates, the
log-likelihood, and the Optim result.

"""
function q5(X, y; startval = zeros(size(X,2)*(maximum(y)-1)))
    K = size(X, 2)
    J = maximum(y)

    alpha_hat_mlogit = fit_mlogit(X, y, startval)
    alpha_mat = reshape(Optim.minimizer(alpha_hat_mlogit), K, J-1)
    loglike = -Optim.minimum(alpha_hat_mlogit)

    println("question 5: multinomial logit estimates from Optim, as the ",
            K, " x ", J-1, " matrix of alpha_j's, rounded to 4 decimals")
    println("(rows: intercept, age, white, collgrad; columns: occupation 1 to ",
            J-1, "; occupation ", J, " is the base):")
    printpretty(round.(alpha_mat, digits=4))
    println("question 5: log-likelihood at the estimates: ",
            round(loglike, digits=4))

    # g_tol = 1e-5 is not always the criterion that stops Optim,
    # so I print which criteria are met
    println("question 5: stopping criteria met -- gradient (g_tol=1e-5): ",
            Optim.g_converged(alpha_hat_mlogit),
            ", objective change: ", Optim.f_converged(alpha_hat_mlogit),
            ", parameter change: ", Optim.x_converged(alpha_hat_mlogit))
    println("question 5: Optim's gradient norm at the estimates: ",
            Optim.g_residual(alpha_hat_mlogit), ", after ",
            Optim.iterations(alpha_hat_mlogit), " iterations")

    return (alpha = alpha_mat, loglike = loglike, result = alpha_hat_mlogit)
end

"""
    q5_checks(X, y, alpha_mat)

Checks the question 5 estimates in three ways:

1. the first-order conditions X'(d - P) are close to zero,
2. the predicted shares match the observed shares,
3. `mlogit_like` and `mlogit_like_loop` give the same value.

Returns `(maxscore, maxsharegap, loopgap)`.

"""
function q5_checks(X, y, alpha_mat)
    N = size(X, 1)
    J = maximum(y)
    alpha = vec(alpha_mat)

    P = mlogit_probs(alpha, X, J)
    d = [y[i] == j for i in 1:N, j in 1:J]

    # 1. first-order conditions for each non-base alternative
    score = X'*(d[:, 1:J-1] .- P[:, 1:J-1])
    maxscore = maximum(abs.(score))

    # 2. predicted shares and observed shares
    observed = vec(sum(d, dims=1))./N
    predicted = vec(sum(P, dims=1))./N
    maxsharegap = maximum(abs.(observed .- predicted))

    # 3. mlogit_like compared with the loop version
    loopgap = abs(mlogit_like(alpha, X, y) - mlogit_like_loop(alpha, X, y))

    println("question 5 check: largest first-order condition X'(d - P) ",
            "at the estimates: ", maxscore)
    println("question 5 check: largest gap between predicted and observed ",
            "choice shares: ", maxsharegap)
    println("question 5 check: gap between mlogit_like and the naive loop ",
            "version: ", loopgap)

    return (maxscore = maxscore, maxsharegap = maxsharegap, loopgap = loopgap)
end

"""
    q5_startvals(X, y, alpha_mat)

Estimates the model again from U[0,1] and U[-1,1] starting values and compares
with `alpha_mat`, the estimates from zeros.

Returns `(unif01, unifpm1)`: the largest difference for each start.

"""
function q5_startvals(X, y, alpha_mat)
    nparams = length(alpha_mat)
    starts = [("U[0,1]",  rand(nparams)),
              ("U[-1,1]", 2 .* rand(nparams) .- 1)]

    gaps = Float64[]
    for (name, startval) in starts
        res = fit_mlogit(X, y, startval)
        gap = maximum(abs.(Optim.minimizer(res) .- vec(alpha_mat)))
        push!(gaps, gap)
        println("question 5 starting values: from a ", name, " vector, ",
                "log-likelihood ", round(-Optim.minimum(res), digits=4),
                ", largest gap from the zero-start estimates ", gap)
    end

    return (unif01 = gaps[1], unifpm1 = gaps[2])
end


#***************************************************
# question 6
#***************************************************
"""
    allwrap()

Answers question 6. Runs all the questions in order. It sets the seed first,
so the random starting values are the same every run.

Returns nothing.

"""
function allwrap()
    Random.seed!(1234)

    #-----------------------------------------------------------------------
    # question 1
    #-----------------------------------------------------------------------
    q1()

    #-----------------------------------------------------------------------
    # questions 2 to 4 use all the observations
    #-----------------------------------------------------------------------
    df = load_nlsw88()
    X = build_X(df)
    y = df.married .== 1

    q2(X, y, df)
    b_logit = q3(X, y)
    q4(df, b_logit)

    #-----------------------------------------------------------------------
    # question 5 uses the cleaned data, which has fewer rows,
    # so I make X and y again
    #-----------------------------------------------------------------------
    dfm = clean_occupation(df)
    Xm = build_X(dfm)
    ym = dfm.occupation

    mnl = q5(Xm, ym)
    q5_checks(Xm, ym, mnl.alpha)
    q5_startvals(Xm, ym, mnl.alpha)

    println("allwrap() finished: every question ran.")

    return nothing
end
