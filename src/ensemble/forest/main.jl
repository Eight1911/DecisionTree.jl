
module Forest

    import Distributed
    import Random
    import ..Struct
    import ..BuildTree
    import ..Misc
    import ..Struct

    function build_forest(
            labels              :: Vector{T},
            features            :: Matrix{S};
            n_trees             = 10,
            partial_sampling    = 0.7,
            max_features        = -1,
            max_depth           = -1,
            min_samples_leaf    = 1,
            min_samples_split   = 2,
            min_purity_increase = 0.0,
            rng                 = Random.GLOBAL_RNG) where {S, T}

        t_samples = length(labels)
        weights   = ones(t_samples)
        BuildTree.Classifier.check_input(
            features, labels, weights,
            max_features,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            min_purity_increase)

        if n_trees < 1
            throw("the number of trees must be >= 1")
        end
        if !(0.0 < partial_sampling <= 1.0)
            throw("partial_sampling must be in (0,1]")
        end

        if max_features == -1
            n_features = size(features, 2)
            max_features = round(Int, sqrt(n_features))
        end

        rng       = Misc.mk_rng(rng)
        n_samples = floor(Int, partial_sampling * t_samples)
        sampler   = BuildTree.Util.sampler(t_samples, n_samples, n_trees)
        list, Y   = BuildTree.Util.assign(labels)
        n_classes = length(list)
        forest    = Array{Struct.Tree{S, T}}(undef, n_trees)
        @sync Distributed.@distributed for i in 1:n_trees
            indX = sampler(rng)
            #=
            root = BuildTree.Classifier._run(
                features, Y, weights,
                indX                = indX,
                loss                = BuildTree.Util.entropy,
                n_classes           = n_classes,
                max_features        = Int(max_features),
                max_depth           = Int(max_depth),
                min_samples_leaf    = Int(min_samples_leaf),
                min_samples_split   = Int(min_samples_split),
                min_purity_increase = Float64(min_purity_increase),
                rng                 = rng)
            # =# 
            root = BuildTree.Classifier._run(
                features, Y, weights,
                indX,
                BuildTree.Util.entropy,
                n_classes,
                Int(max_features),
                Int(max_depth),
                Int(min_samples_leaf),
                Int(min_samples_split),
                Float64(min_purity_increase),
                rng)
            tree = BuildTree.Classifier.Tree(root, list, indX)
            forest[i] = BuildTree.light_classifier(tree, labels, "entropy")
        end

        return Struct.Ensemble{S, T}(forest, nothing, "forest")
    end

    function build_forest(
            labels              :: Vector{T},
            features            :: Matrix{S};
            n_trees             = 10,
            partial_sampling    = 0.7,
            max_features        = -1,
            max_depth           = -1,
            min_samples_leaf    = 1,
            min_samples_split   = 2,
            min_purity_increase = 0.0,
            rng                 = Random.GLOBAL_RNG) where {S, T <: Float64}

        t_samples = length(labels)
        weights   = ones(t_samples)
        BuildTree.Regressor.check_input(
            features, labels, weights,
            max_features,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            min_purity_increase)

        if n_trees < 1
            throw("the number of trees must be >= 1")
        end
        if !(0.0 < partial_sampling <= 1.0)
            throw("partial_sampling must be in the range (0,1]")
        end

        if max_features == -1
            n_features = size(features, 2)
            max_features = round(Int, sqrt(n_features))
        end

        rng       = Misc.mk_rng(rng)
        n_samples = floor(Int, partial_sampling * t_samples)
        sampler   = BuildTree.Util.sampler(t_samples, n_samples, n_trees)
        forest    = Array{Struct.Tree{S, T}}(undef, n_trees)
        # @sync Distributed.@distributed 
        for i in 1:n_trees
            indX = sampler(rng)
            root = BuildTree.Regressor._fit(
                X                   = features, 
                Y                   = labels, 
                W                   = weights,
                indX                = indX,
                loss                = "none",
                max_features        = Int(max_features),
                max_depth           = Int(max_depth),
                min_samples_leaf    = Int(min_samples_leaf),
                min_samples_split   = Int(min_samples_split),
                min_purity_increase = Float64(min_purity_increase),
                rng                 = rng)
            forest[i] = BuildTree.light_regressor(root, nothing, "regression")
        end

        return Struct.Ensemble{S, T}(forest, nothing, "forest")
    end

end

