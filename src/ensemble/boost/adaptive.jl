
# TODO : make a nice interface
module Adaptive
    import ...Tree.Classifier
    import ...Tree.Regressor

    # update the new weights for the next iteration of adaboost
    # using accuracy of the predictions and return the new coefficient 
    function update_coeffs(P, Y, W, n_classes, learning_rate)
        # assume total weight is 1. i.e.
        # W is already normalized so that sum(W) = 1

        err = 0.0
        for i in 1:length(P)
            weight = W[i]
            if P[i] != Y[i]
                err += weight
            end
        end

        nay   = (n_classes - 1) * (1 - err) / max(1e-15, err)
        nay  ^= learning_rate
        alpha = np.log(nay)
        total = (err * nay) + (1.0 - err) # new total weight
        nay   = nay / total # scaling of things that are wrong
        yay   = 1.0 / total # scaling of things that are right
        for i in 1:length(P)
            if P[i] == Y[i]
                W[i] *= yay
            else
                W[i] *= nay
            end
        end

        return err, alpha # the weight for the new tree in the ensemble
    end

    # copy the values fitted to each sample into the array P
    function leaf_vals(node, P)
        if node.is_leaf
            P[node.region] = node.label
        else
            predict(node.l, P)
            predict(node.r, P)
        end
    end

    # SAMME algorithm for multiclass adaboost
    function samme(
            fit           :: Function,
            X             :: Matrix{S},
            Y             :: Vector{T},
            n_estimators  :: Int,
            learning_rate :: Float64;
            rng           = Random.GLOBAL_RNG) where {S, T}
        # TODO : check inputs
        # TODO : make this more efficient (?)
        n_classes = length(Set(Y))
        n_samples = length(Y)
        coeffs    = Float64[]
        trees     = Tree.Classifier.Tree[]
        W         = ones(n_samples) / n_samples
        P         = Array{T}(undef, n_samples)
        for i in 1:n_estimators
            tree       = fit(Y, X, W, rng=rng)
            n_classes  = length(tree.list)
            leaf_vals(tree.root, P)
            err, alpha = update_coeffs(Y, P, W, n_classes, learning_rate)
            push!(coeffs, alpha)
            push!(stumps, tree)
            if err < 1e-7
                break
            end
        end

        return (trees, coeffs)
    end

end
