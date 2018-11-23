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
            n_subfeatures        = -1,
            max_depth            = -1,
            min_samples_leaf     = 1,
            min_samples_split    = 2,
            min_purity_increase  = 0.0;
            rng                  = Random.GLOBAL_RNG) where {S, T}

        tree = Tree.Regressor._fit(
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

        return Misc.light_regressor(t, labels, loss)
    end

end # module