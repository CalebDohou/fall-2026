#*********************************************************
# ECON 6343: Econometrics III
# Problem Set 1: question 5 -- unit tests
# Author:Caleb Dohou
#
# this script reads in the source code and then tests it
#*********************************************************

using Test

include("PS1_Dohou_source.jl")

# q1() writes four files to the working directory, so we run it inside a
# temporary folder to keep the problem set folder clean
tmpdir = mktempdir()
A,B,C,D = cd(q1,tmpdir)

# everything q1() saved, read back off the disk
mp = load(joinpath(tmpdir,"matrixpractice.jld"))
fm = load(joinpath(tmpdir,"firstmatrix.jld"))


#***************************************************
# tests for q1()
#***************************************************
@testset "q1()" begin

    #--------------------------
    # part (a): dimensions and element types
    #--------------------------
    @test size(A) == (10,7)
    @test size(B) == (10,7)
    @test size(C) == (5,7)
    @test size(D) == (10,7)
    @test eltype(A) == Float64
    @test eltype(B) == Float64
    @test eltype(D) == Float64

    #--------------------------
    # part (a): the seed really is 1234 and the draws come off the stream in
    # the stated order and from the stated distributions. redrawing from a
    # freshly seeded stream has to reproduce A and B exactly -- this pins the
    # seed value itself, which a q1()-vs-q1() comparison cannot do
    #--------------------------
    Random.seed!(1234)
    @test A == rand(Uniform(-5,10),10,7)
    @test B == rand(Normal(-2,15),10,7)

    #--------------------------
    # part (a): A is drawn from U[-5,10]
    #--------------------------
    @test all(A .>= -5)
    @test all(A .<= 10)
    # containment alone would also hold for a much narrower distribution, so
    # check that the 70 draws actually cover the support and have the right mean
    @test minimum(A) < -4
    @test maximum(A) > 9
    @test 2.5-4*(15/sqrt(12))/sqrt(70) < mean(A) < 2.5+4*(15/sqrt(12))/sqrt(70)

    #--------------------------
    # part (a): B is drawn from N(-2,15). nothing else in the suite constrains
    # B's distribution, since every other test that mentions B is structural
    #--------------------------
    @test -2-4*15/sqrt(70) < mean(B) < -2+4*15/sqrt(70)
    @test 15/2 < std(B) < 15*2

    #--------------------------
    # part (a): C stacks A and B as described
    #--------------------------
    @test C[:,1:5] == A[1:5,1:5]
    @test C[:,6:7] == B[1:5,6:7]

    #--------------------------
    # part (a): D keeps the nonpositive entries of A and zeroes the rest
    #--------------------------
    @test all(D .<= 0)
    @test all(D[A .> 0] .== 0)
    @test all(D[A .<= 0] .== A[A .<= 0])
    @test count(D .< 0) == count(A .< 0)
    @test count(D .== 0) == count(A .> 0)

    #--------------------------
    # parts (b) and (c): counts
    #--------------------------
    @test length(A) == 70
    # every positive entry of A collapses to the same 0, so the number of
    # unique entries of D is (number of nonpositive draws) + 1
    @test length(unique(D)) == sum(A .<= 0) + 1
    # under seed 1234 that number is 16, which is what the writeup reports
    @test sum(A .<= 0) == 15
    @test length(unique(D)) == 16

    #--------------------------
    # part (a): q1() is reproducible and returns exactly four matrices
    #--------------------------
    out = cd(q1,mktempdir())
    @test length(out) == 4
    A2,B2,C2,D2 = out
    @test A == A2
    @test B == B2
    @test C == C2
    @test D == D2

    #--------------------------
    # parts (h) through (k): the four output files get written
    #--------------------------
    @test isfile(joinpath(tmpdir,"matrixpractice.jld"))
    @test isfile(joinpath(tmpdir,"firstmatrix.jld"))
    @test isfile(joinpath(tmpdir,"Cmatrix.csv"))
    @test isfile(joinpath(tmpdir,"Dmatrix.dat"))

    #--------------------------
    # parts (h) and (i): the .jld files hold exactly the right variables and
    # round-trip without loss. firstmatrix must NOT contain E, F or G
    #--------------------------
    @test sort(collect(keys(fm))) == ["A","B","C","D"]
    @test sort(collect(keys(mp))) == ["A","B","C","D","E","F","G"]
    @test fm["A"] == A
    @test fm["B"] == B
    @test fm["C"] == C
    @test fm["D"] == D

    #--------------------------
    # part (d): E is the vec operator applied to B, i.e. B's columns stacked
    #--------------------------
    @test length(mp["E"]) == 70
    @test mp["E"] == B[:]
    @test mp["E"] == vec(B)
    @test mp["E"] == reshape(B,length(B),1)[:]
    # vec stacks columns, so the first 10 entries are B's first column
    @test mp["E"][1:10] == B[:,1]
    @test mp["E"][11:20] == B[:,2]

    #--------------------------
    # parts (e) and (f): F holds A and B and has been twisted to 2x10x7
    #--------------------------
    @test size(mp["F"]) == (2,10,7)
    @test mp["F"][1,:,:] == A
    @test mp["F"][2,:,:] == B

    #--------------------------
    # part (g): G is the Kronecker product of B and C
    #--------------------------
    @test size(mp["G"]) == (50,49)
    @test mp["G"] == kron(B,C)
    # spot-check the block structure: the top-left block is B[1,1]*C
    @test mp["G"][1:5,1:7] ≈ B[1,1]*C
    @test mp["G"][6:10,1:7] ≈ B[2,1]*C
    # and kron(C,F) really does fail, which is the answer to part (g)
    @test_throws MethodError kron(C,mp["F"])

    #--------------------------
    # part (j): Cmatrix.csv reads back as C with the :auto column names
    #--------------------------
    Cin = CSV.read(joinpath(tmpdir,"Cmatrix.csv"),DataFrame)
    @test size(Cin) == (5,7)
    @test Matrix(Cin) ≈ C
    @test names(Cin) == ["x1","x2","x3","x4","x5","x6","x7"]

    #--------------------------
    # part (k): Dmatrix.dat is genuinely tab-delimited, not comma-delimited
    #--------------------------
    Din = CSV.read(joinpath(tmpdir,"Dmatrix.dat"),DataFrame,delim='\t')
    @test size(Din) == (10,7)
    @test Matrix(Din) ≈ D
    raw = read(joinpath(tmpdir,"Dmatrix.dat"),String)
    @test occursin('\t',raw)
    @test !occursin(',',raw)
end


#***************************************************
# tests for q2()
#***************************************************
@testset "q2()" begin

    #--------------------------
    # part (f): signature and return value
    #--------------------------
    # q2() is required to return nothing, so the only channel it exposes is
    # what it prints. capture that and check it, which does exercise the source
    logfile = joinpath(tmpdir,"q2_stdout.txt")
    ret = open(logfile,"w") do io
        redirect_stdout(io) do
            q2(A,B,C)
        end
    end
    printed = read(logfile,String)
    @test ret === nothing
    # part (a) inside the source: the loop and the broadcast agree
    @test occursin("AB and AB2 agree: true",printed)
    # part (b) inside the source: the loop and the vectorized filter agree
    @test occursin("Cprime and Cprime2 agree: true",printed)
    # part (c)/(e) inside the source: N really is 15,169 and T really is 5
    @test occursin("size of Y is (15169, 5)",printed)

    #--------------------------
    # part (d) inside the source: q2() shows beta, so we can parse the matrix
    # back out of the printed output and check the actual beta the source
    # built -- not a copy of it rebuilt here in the test file
    #--------------------------
    m = match(r"β = (\[[^\]]*\])",printed)
    @test m !== nothing
    beta_src = eval(Meta.parse(m.captures[1]))
    @test size(beta_src) == (6,5)
    @test beta_src[1,:] == [1.0,1.25,1.5,1.75,2.0]
    @test beta_src[2,:] ≈ log.(1:5)
    @test beta_src[3,:] ≈ -sqrt.(1:5)
    @test beta_src[4,:] ≈ [exp(t)-exp(t+1) for t = 1:5]
    @test beta_src[5,:] == collect(1.0:5)
    @test beta_src[6,:] ≈ (1:5)./3

    #--------------------------
    # part (c) inside the source: q2() shows the column means of the X it
    # actually built, so every column's distribution is checked at the source
    #--------------------------
    Nsrc = 15_169
    m = match(r"Xbar = (\[[^\]]*\])",printed)
    @test m !== nothing
    Xbar = eval(Meta.parse(m.captures[1]))
    @test size(Xbar) == (6,5)
    # column 1: intercept
    @test all(Xbar[1,:] .== 1.0)
    # column 2: dummy with probability .75*(6-t)/5, declining over t
    for t = 1:5
        p = 0.75*(6-t)/5
        @test Xbar[2,t] ≈ p atol=4*sqrt(p*(1-p)/Nsrc)
    end
    @test all(diff(Xbar[2,:]) .< 0)
    # column 3: mean 15+t-1 (exact at t=1 where the sd is 0)
    @test Xbar[3,1] ≈ 15.0
    for t = 2:5
        @test Xbar[3,t] ≈ 15+t-1 atol=4*5*(t-1)/sqrt(Nsrc)
    end
    # column 4: mean π(6-t)/3 -- this is the check that catches a missing /3
    for t = 1:5
        @test Xbar[4,t] ≈ pi*(6-t)/3 atol=4*(1/exp(1))/sqrt(Nsrc)
    end
    # columns 5 and 6: binomial means 20*0.6 = 12 and 20*0.5 = 10, stationary
    @test all(Xbar[5,:] .≈ Xbar[5,1])
    @test all(Xbar[6,:] .≈ Xbar[6,1])
    @test Xbar[5,1] ≈ 12.0 atol=4*sqrt(20*0.6*0.4/Nsrc)
    @test Xbar[6,1] ≈ 10.0 atol=4*sqrt(20*0.5*0.5/Nsrc)

    #--------------------------
    # part (e) inside the source: q2() shows the error sd in each period of the
    # Y it actually built. it must be 0.36 in every period, not growing with t
    #--------------------------
    m = match(r"εsd = (\[[^\]]*\])",printed)
    @test m !== nothing
    εsd = eval(Meta.parse(m.captures[1]))
    @test length(εsd) == 5
    for t = 1:5
        @test εsd[t] ≈ 0.36 atol=0.02
    end

    #--------------------------
    # part (a): the loop and the broadcasted version agree
    #--------------------------
    AB = zeros(size(A))
    for r in axes(A,1)
        for c in axes(A,2)
            AB[r,c] = A[r,c]*B[r,c]
        end
    end
    AB2 = A.*B
    @test AB == AB2
    @test size(AB) == (10,7)
    @test AB[3,4] == A[3,4]*B[3,4]

    #--------------------------
    # part (b): the loop and the vectorized filter agree
    #--------------------------
    Cprime = Float64[]
    for c in axes(C,2)
        for r in axes(C,1)
            if C[r,c] >= -5 && C[r,c] <= 5
                push!(Cprime,C[r,c])
            end
        end
    end
    Cprime2 = C[(C.>=-5) .& (C.<=5)]
    @test Cprime == Cprime2
    @test all(Cprime .>= -5)
    @test all(Cprime .<= 5)
    @test length(Cprime) == sum((C.>=-5) .& (C.<=5))
    @test isa(Cprime,Vector{Float64})
    # nothing outside the band survives the filter
    @test !any(abs.(Cprime) .> 5)

    #--------------------------
    # part (c): X has the right shape and every column follows its stated
    # distribution. rebuilt at a smaller N, under its own seed so that the
    # distributional checks below are reproducible rather than flaky
    #--------------------------
    Random.seed!(9876)
    N = 2_000
    K = 6
    T = 5
    X = zeros(N,K,T)
    for i in axes(X,1)
        X[i,1,:] .= 1.0
        X[i,5,:] .= rand(Binomial(20,0.6))
        X[i,6,:] .= rand(Binomial(20,0.5))
        for t in axes(X,3)
            X[i,2,t] = rand() <= 0.75*(6-t)/5
            X[i,3,t] = rand(Normal(15+t-1,5*(t-1)))
            X[i,4,t] = rand(Normal(pi*(6-t)/3,1/exp(1)))
        end
    end
    @test size(X) == (N,K,T)

    # column 1 is an intercept
    @test all(X[:,1,:] .== 1.0)

    # column 2 is a dummy whose success probability is .75*(6-t)/5 and so
    # declines over time
    @test all(in([0.0,1.0]),X[:,2,:])
    for t = 1:T
        p = 0.75*(6-t)/5
        @test mean(X[:,2,t]) ≈ p atol=4*sqrt(p*(1-p)/N)
    end
    @test all(diff([mean(X[:,2,t]) for t = 1:T]) .< 0)

    # column 3 is normal with mean 15+t-1 and standard deviation 5(t-1); at
    # t=1 that standard deviation is 0, so the column is a point mass at 15
    @test all(X[:,3,1] .≈ 15.0)
    for t = 2:T
        @test mean(X[:,3,t]) ≈ 15+t-1 atol=4*5*(t-1)/sqrt(N)
        @test std(X[:,3,t]) ≈ 5*(t-1) rtol=0.15
    end

    # column 4 is normal with mean pi*(6-t)/3 and standard deviation 1/e.
    # the mean check is what catches a missing /3
    for t = 1:T
        @test mean(X[:,4,t]) ≈ pi*(6-t)/3 atol=4*(1/exp(1))/sqrt(N)
        @test std(X[:,4,t]) ≈ 1/exp(1) rtol=0.15
    end

    # columns 5 and 6 are binomial with n=20 and p=0.6 and 0.5 respectively,
    # so they have different means and are integer valued
    @test all(0 .<= X[:,5,1] .<= 20)
    @test all(0 .<= X[:,6,1] .<= 20)
    @test all(X[:,5,1] .== round.(X[:,5,1]))
    @test all(X[:,6,1] .== round.(X[:,6,1]))
    @test mean(X[:,5,1]) ≈ 12.0 atol=4*sqrt(20*0.6*0.4/N)
    @test mean(X[:,6,1]) ≈ 10.0 atol=4*sqrt(20*0.5*0.5/N)
    @test std(X[:,5,1]) ≈ 2.19 rtol=0.15

    # columns 1, 5 and 6 are stationary over time, the others are not
    @test all(X[:,1,1] .== X[:,1,T])
    @test all(X[:,5,1] .== X[:,5,T])
    @test all(X[:,6,1] .== X[:,6,T])
    @test X[:,3,1] != X[:,3,T]
    @test X[:,4,1] != X[:,4,T]

    #--------------------------
    # part (d): beta follows the stated time paths, each checked against its
    # own functional form
    #--------------------------
    beta = zeros(K,T)
    beta[1,:] = [1+0.25*(t-1) for t = 1:T]
    beta[2,:] = [log(t)       for t = 1:T]
    beta[3,:] = [-sqrt(t)     for t = 1:T]
    beta[4,:] = [exp(t)-exp(t+1) for t = 1:T]
    beta[5,:] = [t            for t = 1:T]
    beta[6,:] = [t/3          for t = 1:T]
    @test size(beta) == (K,T)
    @test beta[1,:] == [1.0,1.25,1.5,1.75,2.0]
    @test beta[2,:] ≈ log.(1:T)
    @test beta[3,:] ≈ -sqrt.(1:T)
    @test beta[4,:] ≈ [exp(t)-exp(t+1) for t = 1:T]
    @test beta[5,:] == collect(1.0:T)
    @test beta[6,:] ≈ (1:T)./3
    # row 4 is e^t - e^{t+1}, which is negative and strictly decreasing --
    # a sign check alone would not distinguish it from other negative paths
    @test all(beta[4,:] .< 0)
    @test all(diff(beta[4,:]) .< 0)
    @test beta[2,1] == 0.0
    @test beta[3,:] == -sqrt.(collect(1.0:T))

    #--------------------------
    # part (e): Y is N x T, is generated by X_t*beta_t plus noise, and the
    # noise is N(0,0.36) in every period
    #--------------------------
    Y = hcat([X[:,:,t]*beta[:,t] .+ rand(Normal(0,0.36),N) for t = 1:T]...)
    @test size(Y) == (N,T)

    # with the errors switched off, OLS returns beta_t exactly. t=1 is skipped
    # because column 3 of X is degenerate there, which makes X'X singular
    Y0 = hcat([X[:,:,t]*beta[:,t] for t = 1:T]...)
    for t = 2:T
        @test X[:,:,t]\Y0[:,t] ≈ beta[:,t]
    end

    # the errors are N(0,0.36) in every period: the standard deviation does
    # not grow with t, and it is centered at zero
    resid = hcat([Y[:,t] .- X[:,:,t]*beta[:,t] for t = 1:T]...)
    for t = 1:T
        @test std(resid[:,t]) ≈ 0.36 atol=0.06
        @test mean(resid[:,t]) ≈ 0.0 atol=4*0.36/sqrt(N)
    end
    # an error sd that scaled with t would make the last period's spread much
    # wider than the first's; it does not
    @test std(resid[:,T])/std(resid[:,1]) ≈ 1.0 atol=0.25
end
