

module Boosting

    import ..Struct: Tree, Ensemble

    function _weighted_error(actual::Vector, predicted::Vector, weights::Vector{T}) where T <: Real
        mismatches = actual .!= predicted
        err = sum(weights[mismatches]) / sum(weights)
        return err
    end


    function build_adaboost_stumps(
            labels       :: Vector{T},
            features     :: Matrix{S},
            n_iterations :: Integer;
            rng           = Random.GLOBAL_RNG) where {S, T}
        N = length(labels)
        weights = ones(N) / N
        stumps = Node{S, T}[]
        coeffs = Float64[]
        for i in 1:n_iterations
            new_stump = build_stump(labels, features, weights; rng=rng)
            predictions = apply_tree(new_stump, features)
            err = _weighted_error(labels, predictions, weights)
            new_coeff = 0.5 * log((1.0 + err) / (1.0 - err))
            matches = labels .== predictions
            weights[(!).(matches)] *= exp(new_coeff)
            weights[matches] *= exp(-new_coeff)
            weights /= sum(weights)
            push!(coeffs, new_coeff)
            push!(stumps, new_stump)
            if err < 1e-6
                break
            end
        end
        return Ensemble{S, T}(stumps, coeffs)
    end

    function apply_adaboost_stumps(stumps::Ensemble{S, T}, coeffs::Vector{Float64}, features::Vector{S}) where {S, T}
        n_stumps = length(stumps)
        counts = Dict()
        for i in 1:n_stumps
            prediction = apply_tree(stumps.trees[i], features)
            counts[prediction] = get(counts, prediction, 0.0) + coeffs[i]
        end
        top_prediction = stumps.trees[1].left.majority
        top_count = -Inf
        for (k,v) in counts
            if v > top_count
                top_prediction = k
                top_count = v
            end
        end
        return top_prediction
    end

    function apply_adaboost_stumps(stumps::Ensemble{S, T}, coeffs::Vector{Float64}, features::Matrix{S}) where {S, T}
        n_samples = size(features, 1)
        predictions = Array{T}(undef, n_samples)
        for i in 1:n_samples
            predictions[i] = apply_adaboost_stumps(stumps, coeffs, features[i,:])
        end
        return predictions
    end

    """    apply_adaboost_stumps_proba(stumps::Ensemble, coeffs, features, labels::Vector)

    computes P(L=label|X) for each row in `features`. It returns a `N_row x
    n_labels` matrix of probabilities, each row summing up to 1.

    `col_labels` is a vector containing the distinct labels
    (eg. ["versicolor", "virginica", "setosa"]). It specifies the column ordering
    of the output matrix. """
    function apply_adaboost_stumps_proba(stumps::Ensemble{S, T}, coeffs::Vector{Float64},
                                         features::Vector{S}, labels::Vector{T}) where {S, T}
        votes = [apply_tree(stump, features) for stump in stumps.trees]
        compute_probabilities(labels, votes, coeffs)
    end

    function apply_adaboost_stumps_proba(stumps::Ensemble{S, T}, coeffs::Vector{Float64},
                                        features::Matrix{S}, labels::Vector{T}) where {S, T}
        stack_function_results(row->apply_adaboost_stumps_proba(stumps, coeffs, row, labels), features)
    end

end