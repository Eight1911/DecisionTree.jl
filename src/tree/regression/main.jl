module RegressionTree

    import ..Tree.Regressor
    import ..Misc
    #=
    function build_stump(
            labels   :: Vector{T},
            features :: Matrix{S};
            rng       = Random.GLOBAL_RNG) where {S, T <: Float64}
        return build_tree(labels, features, 0, 1)
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
        tree = Tree.Regressor._fit(
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

        return Misc.light_regressor(t, labels, loss)
    end

end # module