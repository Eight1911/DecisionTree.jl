
function build_forest(
        labels              :: Vector{T},
        features            :: Matrix{S},
        n_subfeatures       = -1,
        n_trees             = 10,
        partial_sampling    = 0.7,
        max_depth           = -1,
        min_samples_leaf    = 5,
        min_samples_split   = 2,
        min_purity_increase = 0.0;
        rng                 = Random.GLOBAL_RNG) where {S, T <: Float64}

    if n_trees < 1
        throw("the number of trees must be >= 1")
    end
    if !(0.0 < partial_sampling <= 1.0)
        throw("partial_sampling must be in the range (0,1]")
    end

    if n_subfeatures == -1
        n_features = size(features, 2)
        n_subfeatures = round(Int, sqrt(n_features))
    end

    t_samples = length(labels)
    n_samples = floor(Int, partial_sampling * t_samples)

    rngs = mk_rng(rng)::Random.AbstractRNG
    forest = Distributed.@distributed (vcat) for i in 1:n_trees
        inds = rand(rngs, 1:t_samples, n_samples)
        build_tree(
            labels[inds],
            features[inds,:],
            n_subfeatures,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            min_purity_increase,
            rng = rngs)
    end

    if n_trees == 1
        return Ensemble{S, T}([forest])
    else
        return Ensemble{S, T}(forest)
    end
end

function build_forest(
        labels              :: Vector{T},
        features            :: Matrix{S},
        n_subfeatures       = -1,
        n_trees             = 10,
        partial_sampling    = 0.7,
        max_depth           = -1,
        min_samples_leaf    = 1,
        min_samples_split   = 2,
        min_purity_increase = 0.0;
        rng                 = Random.GLOBAL_RNG) where {S, T}

    if n_trees < 1
        throw("the number of trees must be >= 1")
    end
    if !(0.0 < partial_sampling <= 1.0)
        throw("partial_sampling must be in the range (0,1]")
    end

    if n_subfeatures == -1
        n_features = size(features, 2)
        n_subfeatures = round(Int, sqrt(n_features))
    end

    t_samples = length(labels)
    n_samples = floor(Int, partial_sampling * t_samples)

    rngs = mk_rng(rng)::Random.AbstractRNG
    forest = Distributed.@distributed (vcat) for i in 1:n_trees
        inds = rand(rngs, 1:t_samples, n_samples)
        build_tree(
            labels[inds],
            features[inds,:],
            n_subfeatures,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            min_purity_increase,
            rng = rngs)
    end

    if n_trees == 1
        return Ensemble{S, T}([forest])
    else
        return Ensemble{S, T}(forest)
    end
end

function apply_forest(forest::Ensemble{S, T}, features::Vector{S}) where {S, T}
    n_trees = length(forest)
    votes = Array{T}(undef, n_trees)
    for i in 1:n_trees
        votes[i] = apply_tree(forest.trees[i], features)
    end

    if T <: Float64
        return mean(votes)
    else
        return majority_vote(votes)
    end
end

function apply_forest(forest::Ensemble{S, T}, features::Matrix{S}) where {S, T}
    N = size(features,1)
    predictions = Array{T}(undef, N)
    for i in 1:N
        predictions[i] = apply_forest(forest, features[i, :])
    end
    return predictions
end

"""    apply_forest_proba(forest::Ensemble, features, col_labels::Vector)

computes P(L=label|X) for each row in `features`. It returns a `N_row x
n_labels` matrix of probabilities, each row summing up to 1.

`col_labels` is a vector containing the distinct labels
(eg. ["versicolor", "virginica", "setosa"]). It specifies the column ordering
of the output matrix. """
function apply_forest_proba(forest::Ensemble{S, T}, features::Vector{S}, labels) where {S, T}
    votes = [apply_tree(tree, features) for tree in forest.trees]
    return compute_probabilities(labels, votes)
end

apply_forest_proba(forest::Ensemble{S, T}, features::Matrix{S}, labels) where {S, T} =
    stack_function_results(row->apply_forest_proba(forest, row, labels),
                           features)
