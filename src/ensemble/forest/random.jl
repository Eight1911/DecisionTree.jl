
module Forest

    function build_forest(
            fit                 :: Function,
            X                   :: Vector{T},
            Y                   :: Matrix{S},
            n_estimators        :: Int;
            rng                 = Random.GLOBAL_RNG) where {S, T}
        rng    = mk_rng(rng)::Random.AbstractRNG
        W      = ones(length(Y))
        forest = Array{}(undef, n_estimators)
        Distributed.@distributed (vcat) for i in 1:n_trees
            forest[i] = fit(X, Y, W, rng=rng)
        end

        return forest
    end

end