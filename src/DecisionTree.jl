
include("struct.jl")
include("tree/main.jl")
include("ensemble/main.jl")
include("measures.jl")

import Random
import .Struct:
    Tree, Leaf, Node, Ensemble,
    is_leaf, depth, print_tree
import .Forest: build_forest
import .Misc: load_data, prune_tree!
import .Apply: apply
import .Measures: confusion_matrix, ConfusionMatrix


#=

    to implement:
        nfoldCV_tree, nfoldCV_forest, nfoldCV_stumps

        build_adaboost_stumps,
        apply_adaboost_stumps,

    might-implement:
        apply_tree_proba
        apply_forest_proba

        apply_adaboost_stumps_proba,
        nfoldCV_stumps
        build_stump

    to interface:
        apply_tree, apply_forest, apply_adaboost (?)

=#


function build_tree(
        labels              :: Vector{T},
        features            :: Matrix{S},
        n_subfeatures        = -1,
        max_depth            = -1,
        min_samples_leaf     = 1,
        min_samples_split    = 2,
        min_purity_increase  = 0.0;
        rng                  = Random.GLOBAL_RNG) where {S, T <: Float64}

    n_samples, _ = size(features)

    tree = BuildTree.Regressor._fit(
        X                   = features,
        Y                   = labels,
        W                   = nothing,
        indX                = collect(1:n_samples),
        # the loss function is just there
        # for interfacing actually used
        loss                = "none",
        max_features        = Int(n_subfeatures),
        max_depth           = Int(max_depth),
        min_samples_leaf    = Int(min_samples_leaf),
        min_samples_split   = Int(min_samples_split),
        min_purity_increase = Float64(min_purity_increase),
        rng                 = Misc.mk_rng(rng))
    
    return BuildTree.light_regressor(tree, labels, "regression")
end

function build_tree(
        labels              :: Vector{T},
        features            :: Matrix{S},
        n_subfeatures        = -1,
        max_depth            = -1,
        min_samples_leaf     = 1,
        min_samples_split    = 2,
        min_purity_increase  = 0.0;
        rng                  = Random.GLOBAL_RNG) where {S, T}

    n_samples, _ = size(features)
    tree = BuildTree.Classifier._fit(
        X                   = features,
        Y                   = labels,
        W                   = nothing,
        indX                = collect(1:n_samples),
        loss                = "entropy",
        max_features        = Int(n_subfeatures),
        max_depth           = Int(max_depth),
        min_samples_leaf    = Int(min_samples_leaf),
        min_samples_split   = Int(min_samples_split),
        min_purity_increase = Float64(min_purity_increase),
        rng                 = Misc.mk_rng(rng))

    return BuildTree.light_classifier(tree, labels, "entropy")
end




X, Y = load_data("digits")

build_tree(Y, X)
@time ens = build_forest(Y, X, 7, 0.7)

P = apply(ens, X)
println(Measures.confusion_matrix(Y, P))
