module Measures

    import Base: show
    import Random
    import LinearAlgebra

    
    struct ConfusionMatrix
        classes  :: Vector
        matrix   :: Matrix{Int}
        accuracy :: Float64
        kappa    :: Float64 # cohen's kappa
    end

    # returns the confusion matrix between actual labels and predictions
    function confusion_matrix(actual::Vector{T}, predicted::Vector{T}) where T

        function assign(a, b)
            n_samples = length(a)
            @assert n_samples == length(b)
            # TODO: make this more memory efficient
            list = sort(unique([a; b]))
            dict = Dict{T, Int}()
            @simd for i in 1:length(list)
                @inbounds dict[list[i]] = i
            end

            _a = Array{Int}(undef, n_samples)
            _b = Array{Int}(undef, n_samples)
            @simd for i in 1:length(a)
                @inbounds _a[i] = dict[a[i]]
                @inbounds _b[i] = dict[b[i]]
            end

            return list, _a, _b
        end

        function main(actual, predicted)

            @assert length(actual) == length(predicted)

            classes, Y, P = assign(actual, predicted)
            n_classes = length(classes)
            n_samples = length(Y)
            mat = zeros(Int, n_classes, n_classes)
            for (y, p) in zip(Y, P)
                mat[y, p] += 1
            end

            t_sum = 0
            @simd for i in 1:n_classes
                y_sum = 0
                p_sum = 0
                @inbounds @simd for j in 1:n_classes
                    y_sum += mat[i, j]
                    p_sum += mat[j, i]
                end
                t_sum += y_sum * p_sum
            end

            accuracy = LinearAlgebra.tr(mat) / n_samples
            diverge  = t_sum / n_samples^2
            kappa    = (accuracy - diverge) / (1.0 - diverge)
            return ConfusionMatrix(classes, mat, accuracy, kappa)
        end

        return main(actual, predicted)
    end


    function show(io::IO, cm::ConfusionMatrix)
        print(io, "Classes:  ")
        show(io, cm.classes)
        print(io, "\nMatrix:   ")
        display(cm.matrix)
        print(io, "\nAccuracy: ")
        show(io, cm.accuracy)
        print(io, "\nKappa:    ")
        show(io, cm.kappa)
    end

end

module CrossValidate
    
    import Statistics
    import Random
    import ..Measures
    import ..BuildTree
    import ..Forest
    import ..Apply
    import ..Misc


    struct nFoldSplit{S, T}
        X    :: Matrix{S}
        Y    :: Vector{T}
        n    :: Int
        indX :: Vector{Int}

        function nFoldSplit(
                X   :: Matrix{S}, 
                Y   :: Vector{T}, 
                n   :: Int, 
                rng :: Random.AbstractRNG = Random.GLOBAL_RNG) where {S, T}
            n_samples, _ = size(X)
            if n_samples != length(Y)
                error("size of X does not match size of Y:"
                    * " $(n_samples) vs $(length(Y))")
            elseif !(n_samples >= n > 0)
                error("split number should be in [n_samples, 0),"
                    * "but have $(n_samples) >= $(n) > 0")
            end

            return new{S, T}(X, Y, n, Random.randperm(rng, n_samples))
        end
    end

    # TODO: optimize this function
    function Base.iterate(data::nFoldSplit{S, T}, num=0) where {S,T}
        if num == data.n
            return nothing
        end

        n_samples, _ = size(data.X)
        start = round(Int,  1+ num*n_samples/data.n)
        stop  = round(Int, (1+num)*n_samples/data.n)

        trainI = Array{Int}(undef, n_samples - (stop - start + 1))
        trainI[1:start-1]   = view(data.indX, 1:(start-1))
        trainI[start:end] = view(data.indX, stop+1:n_samples)
        testI  = data.indX[start:stop]
        testX  = data.X[testI, :]
        testY  = data.Y[testI]
        trainX = data.X[trainI, :]
        trainY = data.Y[trainI]

        return (trainX, trainY, testX, testY), num+1

    end

    function display_comparison(
            labels :: Vector{T},
            preds  :: Vector{T}) where {T}
        @assert length(labels) == length(preds)
        cm = Measures.confusion_matrix(labels, preds)
        println(cm)
        return cm.accuracy
    end

    function display_comparison(
            labels :: Vector{T},
            preds  :: Vector{T}) where {T <: Float64}
        @assert length(labels) == length(preds)
        err  = 0.0
        for (y, p) in zip(labels, preds)
            err += (y - p)^2
        end

        corr = Statistics.cor(labels, preds)
        println("mean squared error:     ", err/length(labels))
        println("correlation coeff:      ", corr)
        println("coeff of determination: ", corr*corr)
        return corr*corr
    end


    function nfoldCV_tree(
            labels              :: Vector{T},
            features            :: Matrix{S},
            n_folds             :: Integer;
            #pruning_purity       = 1.0,
            max_features         = -1,
            max_depth            = -1,
            min_samples_leaf     = 1,
            min_samples_split    = 2,
            min_purity_increase  = 0.0,
            rng                  = Random.GLOBAL_RNG) where {S, T}
        stats = Array{Float64}(undef, n_folds)
        for (i, data) in enumerate(nFoldSplit(features, labels, n_folds))
            trainX, trainY, testX, testY = data
            tree = BuildTree.build_tree(
                trainY, trainX,
                max_features         = max_features,
                max_depth            = max_depth,
                min_samples_leaf     = min_samples_leaf,
                min_samples_split    = min_samples_split,
                min_purity_increase  = min_purity_increase,
                rng                  = rng)
            # Misc.prune_tree!(tree, pruning_purity)
            pred = Apply.apply(tree, testX)

            println("\nFold ", i)
            stats[i] = display_comparison(testY, pred)
        end
        if T <: Float64
            println("mean coeff of determination: ", sum(stats)/n_folds)
        else
            println("mean accuracy: ", sum(stats)/n_folds)
        end

        return stats
    end

    function nfoldCV_forest(
            labels              :: Vector{T},
            features            :: Matrix{S},
            n_folds             :: Integer;
            n_trees              = 10,
            partial_sampling     = 0.7,
            max_features         = -1,
            max_depth            = -1,
            min_samples_leaf     = 1,
            min_samples_split    = 2,
            min_purity_increase  = 0.0,
            rng                  = Random.GLOBAL_RNG) where {S, T}
        stats = Array{Float64}(undef, n_folds)
        for (i, data) in enumerate(nFoldSplit(features, labels, n_folds))
            trainX, trainY, testX, testY = data
            ensemble = Forest.build_forest(
                trainY, trainX,
                n_trees             = n_trees,
                partial_sampling    = partial_sampling,
                max_features        = max_features,
                max_depth           = max_depth,
                min_samples_leaf    = min_samples_leaf,
                min_samples_split   = min_samples_split,
                min_purity_increase = min_purity_increase,
                rng                 = rng)
            # Misc.prune_tree!(tree, pruning_purity)
            pred = Apply.apply(ensemble, testX)

            println("\nFold ", i)
            stats[i] = display_comparison(testY, pred)
        end
        if T <: Float64
            println("mean coeff of determination: ", sum(stats)/n_folds)
        else
            println("mean accuracy: ", sum(stats)/n_folds)
        end

        return stats
    end


end