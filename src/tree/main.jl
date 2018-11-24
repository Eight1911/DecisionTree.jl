
module BuildTree
    include("tree/util.jl")
    include("tree/classifier.jl")
    include("tree/regressor.jl")

    import Random
    import ..Misc
    import ..Struct: Tree, Leaf, Node

    # convert between two tree representations
    function light_regressor(
            tree    :: BuildTree.Regressor.Tree{S},
            labels  :: Union{Nothing, Vector{T}},
            method  :: String) where {S, T <: Float64}

        function recurse(node)
            if node.is_leaf
                return Leaf{Float64}(node.label, node.region)
            else
                l = recurse(node.l)
                r = recurse(node.r)
                return Node{S, Float64}(node.feature, node.threshold, l, r)
            end
        end

        function main(tree, method)
            root = recurse(tree.root)
            if labels == nothing
                return Tree(root, nothing, method)
            else
                labs = labels[tree.labels]
                return Tree(root, labs, method)
            end
        end

        return main(tree, method)
    end

    # convert between two tree representations
    function light_classifier(
            tree    :: BuildTree.Classifier.Tree{S, T},
            labels  :: Union{Nothing, Vector{T}},
            method  :: String) where {S, T}

        function recurse(node, list)
            if node.is_leaf
                label = list[node.label]
                return Leaf{T}(label, node.region)
            else
                l = recurse(node.l, list)
                r = recurse(node.r, list)
                return Node{S, T}(node.feature, node.threshold, l, r)
            end
        end

        function main(tree, labels, method)
            root = recurse(tree.root, tree.list)
            if labels == nothing
                return Tree(root, nothing, method)
            else
                labs = labels[tree.labels]
                return Tree(root, labs, method)
            end
        end

        return main(tree, labels, method)
    end


    function build_tree(
            labels              :: Vector{T},
            features            :: Matrix{S};
            max_features         = -1,
            max_depth            = -1,
            min_samples_leaf     = 1,
            min_samples_split    = 2,
            min_purity_increase  = 0.0,
            rng                  = Random.GLOBAL_RNG) where {S, T <: Float64}

        n_samples, _ = size(features)

        tree = BuildTree.Regressor._fit(
            X                   = features,
            Y                   = labels,
            W                   = nothing,
            indX                = collect(1:n_samples),
            # the loss function is just there for interfacing
            loss                = "none",
            max_features        = Int(max_features),
            max_depth           = Int(max_depth),
            min_samples_leaf    = Int(min_samples_leaf),
            min_samples_split   = Int(min_samples_split),
            min_purity_increase = Float64(min_purity_increase),
            rng                 = Misc.mk_rng(rng))
        
        return BuildTree.light_regressor(tree, labels, "regression")
    end

    function build_tree(
            labels              :: Vector{T},
            features            :: Matrix{S};
            max_features         = -1,
            max_depth            = -1,
            min_samples_leaf     = 1,
            min_samples_split    = 2,
            min_purity_increase  = 0.0,
            rng                  = Random.GLOBAL_RNG) where {S, T}

        n_samples, _ = size(features)
        tree = BuildTree.Classifier._fit(
            X                   = features,
            Y                   = labels,
            W                   = nothing,
            indX                = collect(1:n_samples),
            loss                = "entropy",
            max_features        = Int(max_features),
            max_depth           = Int(max_depth),
            min_samples_leaf    = Int(min_samples_leaf),
            min_samples_split   = Int(min_samples_split),
            min_purity_increase = Float64(min_purity_increase),
            rng                 = Misc.mk_rng(rng))

        return BuildTree.light_classifier(tree, labels, "entropy")
    end

end

# include("classification/main.jl")
# include("regression/main.jl")