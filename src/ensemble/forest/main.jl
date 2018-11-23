
module Forest
    
    import Distributed
    import ..Misc
    import ..Tree.Classifier
    import ..Tree.Regressor
    import ..Tree.Util
    import ..Struct

    function build_forest_classifier(
            labels              :: Vector{T},
            features            :: Matrix{S},
            n_trees             = 10,
            partial_sampling    = 0.7,
            n_subfeatures       = -1,
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

        rngs      = Misc.mk_rng(rng)
        weights   = ones(n_samples)
        t_samples = length(labels)
        n_samples = floor(Int, partial_sampling * t_samples)
        sampler   = Tree.Util.sampler(n_samples, t_samples, n_trees)
        list, Y   = Tree.Util.assign(labels)
        n_classes = length(list)
        forest    = Array{Struct.Tree{S, T}}(undef, n_trees)
        Distributed.@distributed for i in 1:n_trees
            indX = sampler(rng)
            root = Tree.Classifier._run(
                features, Y, weights,
                indX                = indX,
                loss                = Tree.Util.Entropy,
                n_classes           = n_classes,
                max_features        = Int(n_subfeatures),
                max_depth           = Int(max_depth),
                min_samples_leaf    = Int(min_samples_leaf),
                min_samples_split   = Int(min_samples_split),
                min_purity_increase = Float64(min_purity_increase),
                rng                 = Random.GLOBAL_RNG)
            tree = Tree.Classifier.Tree(root, list, indX)
            forest[i] = Misc.light_classifier(tree, labels, "$(Tree.Util.Entropy)")
        end

        return Struct.Ensemble{S, T}(forest, nothing, "classification-forest")
    end

    function build_forest_regressor(
            labels              :: Vector{T},
            features            :: Matrix{S},
            n_trees             = 10,
            partial_sampling    = 0.7,
            n_subfeatures       = -1,
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

        rngs      = Misc.mk_rng(rng)
        weights   = ones(n_samples)
        t_samples = length(labels)
        n_samples = floor(Int, partial_sampling * t_samples)
        sampler   = Tree.Util.sampler(n_samples, t_samples, n_trees)
        forest    = Array{Struct.Tree{S, T}}(undef, n_trees)
        Distributed.@distributed for i in 1:n_trees
            root = Tree.Regressor._fit(
                features, labels, weights,
                indX                = indX,
                max_features        = Int(n_subfeatures),
                max_depth           = Int(max_depth),
                min_samples_leaf    = Int(min_samples_leaf),
                min_samples_split   = Int(min_samples_split),
                min_purity_increase = Float64(min_purity_increase),
                rng                 = Random.GLOBAL_RNG)
            tree = Tree.Regressor.Tree(root, indX)
            forest[i] = Misc.light_regressor(tree, labels, "$(Tree.Util.Entropy)")
        end

        return Struct.Ensemble{S, T}(forest, nothing, "regression-forest")
    end

end

