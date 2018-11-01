

module ClassificationTree
    
    import ..Tree.Util
    import ..Tree.Classifier
    import ..Struct
    import ..Misc
    #=
    function build_stump(
            labels      :: Vector{T},
            features    :: Matrix{S},
            weights      = nothing;
            rng          = Random.GLOBAL_RNG) where {S, T}

        t = Tree.Classifier._fit(
            X                   = features,
            Y                   = labels,
            W                   = weights,
            loss                = "zero_one",
            max_features        = size(features, 2),
            max_depth           = 1,
            min_samples_leaf    = 1,
            min_samples_split   = 2,
            min_purity_increase = 0.0;
            rng                 = rng)

        return _convert(t.root, t.list, labels[t.labels])
    end
    =#

    function build_tree(
            labels              :: Vector{T},
            features            :: Matrix{S},
            n_subfeatures        = 0,
            max_depth            = -1,
            min_samples_leaf     = 1,
            min_samples_split    = 2,
            min_purity_increase  = 0.0;
            rng                  = Random.GLOBAL_RNG) where {S, T}

        if max_depth == -1
            max_depth = typemax(Int)
        end
        if n_subfeatures == 0
            n_subfeatures = size(features, 2)
        end

        rng = Misc.mk_rng(rng)
        loss = "entropy"
        tree = Tree.Classifier._fit(
            X                   = features,
            Y                   = labels,
            W                   = nothing,
            loss                = loss,
            max_features        = Int(n_subfeatures),
            max_depth           = Int(max_depth),
            min_samples_leaf    = Int(min_samples_leaf),
            min_samples_split   = Int(min_samples_split),
            min_purity_increase = Float64(min_purity_increase),
            rng                 = rng)

        return Misc.light_classifier(t, labels, loss)
    end

end # module


