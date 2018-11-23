
# TODO : make a nice interface
module Adaptive
    import ...BuildTree.Classifier
    import ...Struct
    import ...Misc

    # update the new weights for the next iteration of adaboost
    # using accuracy of the predictions and return the new coefficient 
    function update_coeffs(P, Y, W, n_classes, learning_rate)
        # assume total weight is 1. i.e.
        # W is already normalized so that sum(W) = 1
        err = 0.0
        # @inbounds @simd 
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
        # @inbounds @simd
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
    function leaf_vals(tree, P)

        function main(node, indX, P)
            if node.is_leaf
                @inbounds @simd for i in node.region
                    P[indX[i]] = node.label
                end
            else
                main(node.l, P)
                main(node.r, P)
            end
        end

        return main(tree.root, tree.labels, P)
    end

    # SAMME algorithm for multiclass adaboost
    function samme(
            fit           :: Function,
            Y             :: Vector{T},
            n_estimators  :: Int,
            learning_rate :: Float64;
            rng           = Random.GLOBAL_RNG) where {S, T}
        # TODO : check inputs
        rng       = Misc.mk_rng(rng)
        n_samples = length(Y)
        coeffs    = Float64[]
        trees     = BuildTree.Classifier.Tree[]
        W         = ones(n_samples) / n_samples
        P         = Array{T}(undef, n_samples)
        for i in 1:n_estimators
            tree       = fit(W, rng=rng)
            n_classes  = length(tree.list)
            leaf_vals(tree, P)
            err, alpha = update_coeffs(Y, P, W, n_classes, learning_rate)
            push!(coeffs, alpha)
            push!(trees, tree)
            # TODO : add custom tolerance (?)
            if err < 1e-7
                break
            end
        end

        return (trees, coeffs)
    end


    function fitter(
            X                   :: Matrix{S},
            Y                   :: Vector{T},
            loss                = "zero_one",
            partial_sampling    = 0.7,
            max_features        = -1,
            max_depth           = -1,
            min_samples_leaf    = 1,
            min_samples_split   = 2,
            min_purity_increase = 0.0) where {S, T}

        t_samples, n_features = size(X)
        n_samples = floor(partial_sampling * t_samples)
        list, _Y  = Util.assign(Y)
        shuffle   = shuffler(t_samples, n_samples)
        n_classes = length(list)

        # TODO: check partial_sampling
        BuildTree.Classifier.check_input(
            X, _Y, fill(1.0, t_samples),
            max_features,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            min_purity_increase)

        if isa(loss, String)
            if loss in keys(LOSS_DICT)
                loss = LOSS_DICT[loss]
            else
                throw("loss function not supported")
            end
        end

        function main(W; rng)

            if length(W) != n_samples
                throw("dimension mismatch between X"
                    * " and W ($(size(X)) vs $(size(W))")
            end

            indX = shuffle(rng)

            return BuildTree.Classifier._run(
                X, _Y, W,
                indX, loss,
                n_classes,
                max_features,
                max_depth,
                min_samples_leaf,
                min_samples_split,
                min_purity_increase,
                rng=rng)
        end

        return (list, _Y, main)
    end

    function build_adaboost(
        labels              :: Vector{T},
        features            :: Matrix{S},
        n_trees             = 10,
        l_rate              = 0.05,
        partial_sampling    = 0.7,
        n_subfeatures       = -1,
        max_depth           = -1,
        min_samples_leaf    = 1,
        min_samples_split   = 2,
        min_purity_increase = 0.0;
        rng                 = Random.GLOBAL_RNG) where {S, T}

        X = features
        Y = labels
        max_features = n_subfeatures
        loss = "zero_one"
        list, _Y, fit = fitter(
            X, Y,
            loss,
            partial_sampling,
            max_features,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            min_purity_increase)

        trees, coeffs = samme(fit, _Y, n_trees, l_rate, rng=rng)


        return Misc.light_classfier # pause
    end
end
