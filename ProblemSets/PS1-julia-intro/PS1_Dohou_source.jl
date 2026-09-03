#*************************************************
# ECON 6343: Econometrics III
# Problem Set 1: source code (functions only)
# Author: Caleb Dohou
#*************************************************

#   import Pkg; Pkg.add("JLD")
#   import Pkg; Pkg.add("CSV")
#   import Pkg; Pkg.add("DataFrames")
#   import Pkg; Pkg.add("FreqTables")

using JLD, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions

#***************************************************
# question 1
#***************************************************
"""
    q1()

Answers question 1. Takes no inputs. Sets the random seed to 1234 and creates
the matrices `A`, `B`, `C`, `D`, `E`, `F` and `G` described in parts (a)
through (g), reports the number of elements of `A` and the number of unique
elements of `D`, and writes the four output files `matrixpractice.jld`,
`firstmatrix.jld`, `Cmatrix.csv` and `Dmatrix.dat` to the working directory.

Returns the matrices `A`, `B`, `C` and `D`.

"""
function q1()
    #----------------------------------------------------------------------------------------
    # question 1, part (a)
    #----------------------------------------------------------------------------------------
    #  set seed
    Random.seed!(1234)

    # draw 10x7 array of uniform numbers U[-5,10]
    A = rand(Uniform(-5,10), 10, 7)

    # draw 10x7 array of normal numbers N(-2,15)
    B = rand(Normal(-2,15), 10, 7)

    # Indexing: first 5 rows and 5 columns of A, last 2 columns and first 5 rows of B
    C = [A[1:5, 1:5] B[1:5, end-1:end]]

    # BitArray /dummy var: D = A where A <= 0, and 0 otherwise
    D = A .* (A .<= 0)

    #----------------------------------------------------------------------------------------
    # question 1, part (b)
    #----------------------------------------------------------------------------------------
    # number of elements of A
    println("number of elements of A is ",length(A))

    #----------------------------------------------------------------------------------------
    # question 1, part (c)
    #----------------------------------------------------------------------------------------
    # number of unique elements of D
    println("number of unique elements of D is ",length(unique(D)))

    #----------------------------------------------------------------------------------------
    # question 1, part (d)
    #----------------------------------------------------------------------------------------
    # vec operator applied to B using reshape()
    E = reshape(B, length(B), 1)
    # the easier way to do this
    E = B[:]

    #----------------------------------------------------------------------------------------
    # question 1, part (e)
    #----------------------------------------------------------------------------------------
    #3-d array with A in the first slice and B in the second
    F = cat(A, B; dims=3)

    #----------------------------------------------------------------------------------------
    # question 1, part (f)
    #----------------------------------------------------------------------------------------
    # twist F from 10x7x2 to 2x10x7
    F = permutedims(F, (3, 1, 2))

    #----------------------------------------------------------------------------------------
    # question 1, part (g)
    #----------------------------------------------------------------------------------------
    # Kron of B and C
    G = kron(B,C)

    # kron(C,F) produces a MethodError: kron is only defined for vectors and
    # matrices, and F is a 3-dimensional array
    try
        kron(C,F)
    catch e
        println("kron(C,F) fails with a ",typeof(e),": kron is only defined ",
                "for vectors and matrices, and F is 3-dimensional")
    end

    #----------------------------------------------------------------------------------------
    # question 1, part (h)
    #----------------------------------------------------------------------------------------
    save("matrixpractice.jld", "A", A, "B", B, "C", C, "D", D, "E", E, "F", F, "G", G)

    #----------------------------------------------------------------------------------------
    # question 1, part (i)
    #----------------------------------------------------------------------------------------
    save("firstmatrix.jld", "A", A, "B", B, "C", C, "D", D)

    #----------------------------------------------------------------------------------------
    # question 1, part (j)
    #----------------------------------------------------------------------------------------
    CSV.write("Cmatrix.csv", DataFrame(C, :auto))

    #----------------------------------------------------------------------------------------
    # question 1, part (k)
    #----------------------------------------------------------------------------------------
    CSV.write("Dmatrix.dat", DataFrame(D, :auto), delim="\t")

    #----------------------------------------------------------------------------------------
    # question 1, part (l)
    #----------------------------------------------------------------------------------------
    return A, B, C, D
end


#***************************************************
# question 2
#***************************************************
"""
    q2(A,B,C)

Answers question 2 for the matrices `A`, `B` and `C` returned by [`q1`](@ref).
Computes the element-by-element product of `A` and `B` with and without a
loop, collects the elements of `C` lying in [-5,5] with and without a loop,
simulates the panel `X` of dimension N x K x T, builds the coefficient matrix
`β` whose rows evolve over time, and forms `Y` from `X`, `β` and an iid
N(0,0.36) error.

Returns nothing.

"""
function q2(A, B, C)
    #----------------------------------------------------------------------------------------
    # question 2, part (a)
    #----------------------------------------------------------------------------------------
    # element-by-element product of A and B with a loop
    AB = zeros(size(A))
    for r in axes(A,1)
        for c in axes(A,2)
            AB[r,c] = A[r,c] * B[r,c]
        end
    end

    # same but without a loop or comprehension
    AB2 = A .* B
    println("AB and AB2 agree: ",isequal(AB,AB2))

    #----------------------------------------------------------------------------------------
    # question 2, part (b)
    #----------------------------------------------------------------------------------------
    # elements of C between -5 and 5 (inclusive) with a loop
    Cprime = Float64[]
    for c in axes(C,2)
        for r in axes(C,1)
            if C[r,c] >= -5 && C[r,c] <= 5
                push!(Cprime, C[r,c])
            end
        end
    end

    # same but without a loop
    Cprime2 = C[(C .>=-5) .& (C .<= 5)]
    @show Cprime
    println("Cprime and Cprime2 agree: ",isequal(Cprime,Cprime2))

    #----------------------------------------------------------------------------------------
    # question 2, part (c)
    #----------------------------------------------------------------------------------------
    N = 15_169
    K = 6
    T = 5
    X = zeros(N, K, T)

    # column 1: intercept
    # column 2: dummy variable, 1 with probability .75*(6-t)/5
    # column 3: normal with mean 15+t-1 and standard deviation 5(t-1)
    # column 4: normal with mean π(6-t)/3 and standard deviation 1/e
    # column 5: binomial n=20, p=0.6
    # column 6: binomial n=20, p=0.5
    # columns 1, 5 and 6 are stationary over time

    for i in axes(X,1)
        X[i, 1, :] .= 1.0
        X[i, 5, :] .= rand(Binomial(20, 0.6))
        X[i, 6, :] .= rand(Binomial(20, 0.5))
        for t in axes(X,3)
            X[i, 2, t] = rand() <= .75*(6-t)/5
            X[i, 3, t] = rand(Normal(15 + t - 1, 5*(t-1)))
            X[i, 4, t] = rand(Normal(π*(6-t)/3, 1/exp(1)))
        end
    end
    # K x T matrix of column means, one column per period
    Xbar = dropdims(mean(X, dims=1), dims=1)
    @show Xbar

    #----------------------------------------------------------------------------------------
    # question 2, part (d)
    #----------------------------------------------------------------------------------------
    # comprehensions
    β = zeros(K, T)
    β[1, :] = [1 + 0.25*(t-1) for t in 1:T]
    β[2, :] = [log(t) for t in 1:T]
    β[3, :] = [-sqrt(t) for t in 1:T]
    β[4, :] = [exp(t) - exp(t+1) for t in 1:T]
    β[5, :] = [t for t in 1:T]
    β[6, :] = [t/3 for t in 1:T]
    @show β

    #----------------------------------------------------------------------------------------
    # question 2, part (e)
    #----------------------------------------------------------------------------------------
    # Y_t = X_t*β_t + ε_t with ε_t ~ iid N(0, 0.36), built with a comprehension
    Y = hcat([X[:, :, t] * β[:, t] .+ rand(Normal(0, 0.36), N) for t in 1:T]...)
    println("size of Y is ",size(Y))
    # standard deviation of the error in each period
    εsd = [std(Y[:, t] .- X[:, :, t] * β[:, t]) for t in 1:T]
    @show εsd

    #----------------------------------------------------------------------------------------
    # question 2, part (f)
    #----------------------------------------------------------------------------------------
    return nothing
end
