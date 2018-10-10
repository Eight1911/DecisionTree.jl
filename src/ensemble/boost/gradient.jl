module Gradient
    include("gradloss.jl")
    import ...Tree.Classifier
    import ...Tree.Regressor

    # set the label of each leaf to the minimizer
    # of the given loss of the things that fall into it
    # then update the cumulative predictions P,
    # i.e., add the predictions of each leaf of the tree to P.
    function update_step(node, Y, P, R, minimize, learning_rate)
        if node.is_leaf
            node.label = learning_rate * minimize(R, node.region)
            P[node.region] .+= node.label
        else
            minimize_loss(node.l, Y, P, R, minimize, learning_rate)
            minimize_loss(node.r, Y, P, R, minimize, learning_rate)
        end
    end

    function gradient(
            fit           :: Function,
            X             :: Matrix{S},
            Y             :: Vector{T},
            W             :: Vector{U},
            loss          :: GradientLossFunction.Loss,
            n_estimators  :: Int,
            learning_rate :: Float64;
            rng           = Random.GLOBAL_RNG) where {S, T}
        # TODO : check inputs
        # W = ones(n_samples) # weights to make api consistent
        n_samples = length(Y)
        gradient  = loss.minus_gradient
        minimize  = loss.point_minimize
        trees     = Tree.Regressor.Tree[]
        P         = zeros(T, n_samples) # cumulative predictions
        L         = Y.copy()            # Y - P
        R         = Y.copy()            # pseudo-residuals
        eta       = 1.0                 # the learning rate
        for i in 1:(n_iterations - 1)
            # fit the tree to the pseudo residuals
            tree = fit(X, R, W, rng=rng)
            R  .= Y
            R .-= P # R = Y - P
            # update_step might mutate R
            update_step(tree.root, P, R, W, minimize, learning_rate)
            trees.append(tree)
            # after the first iteration,
            # set eta to given learning rate
            eta = learning_rate
            # no need to compute gradients
            # for the last iteration
            for i in 1:n_samples
                R[i] = gradient(Y[i], P[i]) # - dL(Y, P) / dP
            end
        end

        let # run this in the last iteration
            # without calculating the gradients
            tree = fit(X, R, W, rng=rng)
            R  .= Y
            R .-= P # R = Y - P
            # update_step might mutate R
            update_step(tree.root, P, R, W, minimize, learning_rate)
            trees.append(tree)
        end

        return trees
    end

    function fitter(
            max_features        = 100,
            max_depth           = typemax(Int),
            min_samples_leaf    = 1,
            min_samples_split   = 2,
            min_purity_increase = 0.0)

        function run(X, Y, W, rng)
            if max_features == 0
                max_features = shape(X, 2) # n_features
            end
            return Tree.Regressor.fit(
                X, Y, Z,
                max_features,
                max_depth,
                min_samples_leaf,
                min_samples_split,
                min_purity_increase,
                rng=rng)
        end

        return run
    end

end