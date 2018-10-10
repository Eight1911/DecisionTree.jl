# Utilities

## Helper function. Counts the votes.
## Returns a vector of probabilities (eg. [0.2, 0.6, 0.2]) which is in the same
## order as get_labels(classifier) (eg. ["versicolor", "setosa", "virginica"])
function compute_probabilities(labels::Vector, votes::Vector, weights=1.0)
    label2ind = Dict(v => k for (k, v) in enumerate(labels))
    counts = zeros(Float64, length(label2ind))
    for (i, label) in enumerate(votes)
        if isa(weights, Number)
            counts[label2ind[label]] += weights
        else
            counts[label2ind[label]] += weights[i]
        end
    end
    return counts / sum(counts) # normalize to get probabilities
end

# Applies `row_fun(X_row)::Vector` to each row in X
# and returns a Matrix containing the resulting vectors, stacked vertically
function stack_function_results(row_fun::Function, X::Matrix)
    N = size(X, 1)
    N_cols = length(row_fun(X[1, :])) # gets the number of columns
    out = Array{Float64}(undef, N, N_cols)
    for i in 1:N
        out[i, :] = row_fun(X[i, :])
    end
    return out
end


function _convert(
        node   :: Tree.Classifier.NodeMeta{S},
        list   :: Vector{T},
        labels :: Vector{T}) where {S, T}

    if node.is_leaf
        return Leaf{T}(list[node.label], labels[node.region])
    else
        left = _convert(node.l, list, labels)
        right = _convert(node.r, list, labels)
        return Node{S, T}(node.feature, node.threshold, left, right)
    end
end

################################################################################

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

    rng = mk_rng(rng)::Random.AbstractRNG
    t = Tree.Classifier._fit(
        X                   = features,
        Y                   = labels,
        W                   = nothing,
        loss                = "entropy",
        max_features        = Int(n_subfeatures),
        max_depth           = Int(max_depth),
        min_samples_leaf    = Int(min_samples_leaf),
        min_samples_split   = Int(min_samples_split),
        min_purity_increase = Float64(min_purity_increase),
        rng                 = rng)

    return _convert(t.root, t.list, labels[t.labels])
end

function prune_tree(tree::LeafOrNode{S, T}, purity_thresh=1.0) where {S, T}
    if purity_thresh >= 1.0
        return tree
    end
    function _prune_run(tree::LeafOrNode{S, T}, purity_thresh::Real) where {S, T}
        N = length(tree)
        if N == 1        ## a Leaf
            return tree
        elseif N == 2    ## a stump
            all_labels = [tree.left.values; tree.right.values]
            majority = majority_vote(all_labels)
            matches = findall(all_labels .== majority)
            purity = length(matches) / length(all_labels)
            if purity >= purity_thresh
                return Leaf{T}(majority, all_labels)
            else
                return tree
            end
        else
            return Node{S, T}(tree.featid, tree.featval,
                        _prune_run(tree.left, purity_thresh),
                        _prune_run(tree.right, purity_thresh))
        end
    end
    pruned = _prune_run(tree, purity_thresh)
    while length(pruned) < length(tree)
        tree = pruned
        pruned = _prune_run(tree, purity_thresh)
    end
    return pruned
end


apply_tree(leaf::Leaf{T}, feature::Vector{S}) where {S, T} = leaf.majority

function apply_tree(tree::Node{S, T}, features::Vector{S}) where {S, T}
    if features[tree.featid] < tree.featval
        return apply_tree(tree.left, features)
    else
        return apply_tree(tree.right, features)
    end
end

function apply_tree(tree::LeafOrNode{S, T}, features::Matrix{S}) where {S, T}
    N = size(features,1)
    predictions = Array{T}(undef, N)
    for i in 1:N
        predictions[i] = apply_tree(tree, features[i, :])
    end
    if T <: Float64
        return Float64.(predictions)
    else
        return predictions
    end
end

"""    apply_tree_proba(::Node, features, col_labels::Vector)

computes P(L=label|X) for each row in `features`. It returns a `N_row x
n_labels` matrix of probabilities, each row summing up to 1.

`col_labels` is a vector containing the distinct labels
(eg. ["versicolor", "virginica", "setosa"]). It specifies the column ordering
of the output matrix. """
apply_tree_proba(leaf::Leaf{T}, features::Vector{S}, labels) where {S, T} =
    compute_probabilities(labels, leaf.values)

function apply_tree_proba(tree::Node{S, T}, features::Vector{S}, labels) where {S, T}
    if tree.featval === nothing
        return apply_tree_proba(tree.left, features, labels)
    elseif features[tree.featid] < tree.featval
        return apply_tree_proba(tree.left, features, labels)
    else
        return apply_tree_proba(tree.right, features, labels)
    end
end

apply_tree_proba(tree::LeafOrNode{S, T}, features::Matrix{S}, labels) where {S, T} =
    stack_function_results(row->apply_tree_proba(tree, row, labels), features)
