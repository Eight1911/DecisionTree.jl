
# TODO : make a nice interface
module Adaptive
    import Random

    import ...Struct: Tree, Ensemble
    import ...BuildTree
    import ...Misc


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
        n_samples = round(Int, partial_sampling*t_samples)
        shuffler  = BuildTree.Util.shuffler(t_samples, n_samples)
        list, _Y  = BuildTree.Util.assign(Y)
        n_classes = length(list)

        if !(0.0 < partial_sampling <= 1.0)
            throw("partial_sampling must be in (0,1]")
        elseif isa(loss, String)
            if loss in keys(BuildTree.Classifier.LOSS_DICT)
                loss = BuildTree.Classifier.LOSS_DICT[loss]
            else
                throw("loss function not supported")
            end
        end

        BuildTree.Classifier.check_input(
            X, _Y, fill(1.0, t_samples),
            max_features,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            min_purity_increase)


        function main(W; rng)
            if length(W) != t_samples
                throw("dimension mismatch between X"
                    * " and W ($(size(X)) vs $(size(W))")
            end

            indX = shuffler(rng)
            return indX, BuildTree.Classifier._run(
                X, _Y, W,
                indX, loss,
                n_classes,
                max_features,
                max_depth,
                min_samples_leaf,
                min_samples_split,
                min_purity_increase,
                rng)
        end

        return list, _Y, main
    end

    # update the new weights for the next iteration of adaboost
    # using accuracy of the predictions and return the new coefficient 
    function update_coeffs(P, Y, W, n_classes, learning_rate)
        # assume total weight is 1. i.e.
        # W is already normalized so that sum(W) = 1
        err = 0.0
        W_sum = 0.0
        # @inbounds @simd 
        for i in 1:length(P)
            weight = W[i]
            W_sum += weight
            if P[i] != Y[i]
                err += weight
            end
        end

        nay   = (n_classes - 1) * (1 - err) / max(1e-15, err)
        nay  ^= learning_rate
        alpha = log(nay)
        total = (err * nay) + (W_sum - err) # new total weight
        total/= W_sum
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

    function predict(root, indX, X, P)

        function _apply(node, X, i)
            curr = node
            while !curr.is_leaf
                if X[i, curr.feature] < curr.threshold
                    curr = curr.l
                else
                    curr = curr.r
                end
            end

            return curr.label
        end

        function solve_known(node, indX, P)
            if node.is_leaf
                @inbounds @simd for i in node.region
                    P[indX[i]] = node.label
                end
            else
                solve_known(node.l, indX, P)
                solve_known(node.r, indX, P)
            end
        end

        function solve_unknown(node, X, P)
            n_samples = size(X, 1)
            for i in 1:n_samples
                if P[i] > -1
                    continue
                else
                    P[i] = _apply(node, X, i)
                end
            end
        end

        function main(root, indX, X, P)
            P[:] .= -1

            solve_known(root, indX, P)
            solve_unknown(root, X, P)
        end

        return main(root, indX, X, P)
    end

    # SAMME algorithm for multiclass adaboost
    function samme(
            fit           :: Function,
            X             :: Matrix{S},
            Y             :: Vector{Int},
            list          :: Vector{T},
            n_estimators  :: Int,
            learning_rate :: Float64;
            rng           = Random.GLOBAL_RNG) where {S, T}
        # TODO : check inputs
        rng       = Misc.mk_rng(rng)
        n_samples = length(Y)
        n_classes = length(list)
        coeffs    = Float64[]
        trees     = BuildTree.Classifier.Tree[]
        W         = fill(1/n_samples, n_samples)
        P         = Array{T}(undef, n_samples) # predictions
        for i in 1:n_estimators
            indX, root = fit(W, rng=rng)
            predict(root, indX, X, P)
            err, alpha = update_coeffs(Y, P, W, n_classes, learning_rate)
            tree       = BuildTree.Classifier.Tree(root, list, indX)
            push!(coeffs, alpha)
            push!(trees, tree)
            # TODO : add custom tolerance (?)
            # if err < 1e-7
            #     break
            # end
        end

        return trees, coeffs
    end

    function build_adaboost(
        labels              :: Vector{T},
        features            :: Matrix{S},
        n_trees             = 10,
        l_rate              = 0.2;
        partial_sampling    = 0.7,
        max_features        = -1,
        max_depth           = -1,
        min_samples_leaf    = 1,
        min_samples_split   = 2,
        min_purity_increase = 0.0,
        rng                 = Random.GLOBAL_RNG) where {S, T}

        X = features
        Y = labels
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

        trees, coeffs = samme(fit, X, _Y, list, n_trees, l_rate, rng=rng)
        light_trees = Array{Tree{S, T}}(undef, length(trees))
        for (i, t) in enumerate(trees)
            light_trees[i] = BuildTree.light_classifier(t, nothing, loss)
        end

        return Ensemble(light_trees, coeffs, "adaboost")
    end
end
