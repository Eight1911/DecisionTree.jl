
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
        nfoldCV_stumps
        build_adaboost,
        apply_adaboost,

    might-implement:
        apply_proba

    to interface:
        apply_tree, apply_forest, apply_adaboost (?)

=#

import Random
X, Y = load_data("digits")
build_tree(Y, X)
build_forest(Y, X, n_trees=7, partial_sampling=0.7)
@time ens = build_forest(Y, X, n_trees=50, partial_sampling=0.5)

P = apply(ens, X)
import .CrossValidate
CrossValidate.nfoldCV_forest(
    Y, X, 3; n_trees = 150, partial_sampling=0.5)
@time output = CrossValidate.nfoldCV_forest(
    Y, X, 3; n_trees = 150, partial_sampling=0.5)
println(output)
