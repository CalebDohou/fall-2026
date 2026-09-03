#***************************************************
# ECON 6343: Econometrics III
# Problem Set 1: script (runs the source code)
# Author:Caleb Dohou
#***************************************************

include("PS1_Dohou_source.jl")

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 1
#:::::::::::::::::::::::::::::::::::::::::::::::::::
A,B,C,D = q1()

# writeup for question 1:
#
# (b) A has 70 elements.
#
# (c) D has 70 elements with only 16 that are unique. Every entry of A that is
#     positive is linked to the same value, 0. 
#     Exactly 15 out of the 70 draws are non-positive, which
#     gives 15 distinct negative values plus the single 0.
#
# (d) reshape(B,length(B),1) stores the columns of B into a 70x1 matrix. The
#     easier way is B[:], that returns the same numbers as a 
#     70-element vector.
#
#
# (g) G = kron(B,C) is 50x49, since B is 10x7 and C is 5x7 and the Kronecker
#     product multiplies the dimensions. Trying kron(C,F) throws a MethodError:
#     after question 1, part (f) F is a 3-dimensional 2x10x7 array, and kron is only
#     defined for vectors and matrices, so no method matches the argument
#     types. The error is caught and printed rather than left to halt q1().


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::
q2(A,B,C)

# writeup for question 2:
#
# (a) the loop and the broadcasted version A.*B give identical results.
#
# (b) the loop pushes elements column by column, which is the order Julia
#     stores them in, so Cprime matches the vectorized Cprime2 exactly.
#
# (c) column 3 of X is specified to have standard deviation 5(t-1), which is 0
#     at t=1. Normal() accepts a zero standard deviation and returns a point
#     mass, so the spec is coded literally and column 3 is degenerate at its
#     mean of 15 in the first period.
#
# (e) Y is 15169 x 5. The errors are N(0,0.36) in every period: the standard
#     deviation is constant and does not grow with t.
#
# (f) q2() returns nothing, as the problem set requires.
