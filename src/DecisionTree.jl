
include("struct.jl")
include("tree/main.jl")
include("ensemble/main.jl")
include("measures.jl")

import .Struct:
    Tree, Leaf, Node, Ensemble,
    is_leaf, depth, print_tree

import .BuildTree:
    build_tree

import .Forest:
    build_forest

import .Boost.Adaptive:
    build_adaboost

import .Apply:
    apply

import .Misc:
    load_data, prune_tree!

import .Measures:
    confusion_matrix, ConfusionMatrix

import .CrossValidate: 
    nfoldCV_tree, nfoldCV_forest


#=

    to implement:
        nfoldCV_adaboost,
        build_gradientboost
        apply_gradientboost
        nfoldCV_gradientboost

    might-implement:
        apply_proba

    to interface:
        apply_tree, apply_forest, apply_adaboost (?)

=#

import Random
X, Y = load_data("digits")

build_adaboost(Y, X, 7)
build_tree(Y, X)

@time ens = build_adaboost(Y, X, 150, partial_sampling=0.1)
acc = sum(Y .== apply(ens, X))/length(Y)
println("accuracy ", acc)

for i in ens.coeffs
    println(i)
end