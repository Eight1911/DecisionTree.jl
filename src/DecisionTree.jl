include("struct.jl")
include("misc.jl")
include("tree/main.jl")
include("ensemble/main.jl")


import Struct: Leaf, Tree, Node, Ensemble, is_leaf, depth

#=
    exported: Leaf, Node, Tree, Ensemble, is_leaf

    to export: 
        print_tree, depth, 
        build_tree,
        build_forest
        prune_tree,

    to implement:
        apply_forest, 
        build_adaboost_stumps,
        apply_adaboost_stumps,
        ConfusionMatrix, 
        confusion_matrix, 
        load_data

        nfoldCV_tree, nfoldCV_forest, nfoldCV_stumps

    might-implement:
        apply_tree_proba
        apply_forest_proba

        apply_adaboost_stumps_proba,
        nfoldCV_stumps
        build_stump

=#