


module Misc

    import Random
    import ..Struct: Tree, Leaf, Node, is_leaf

    function load_data(name)
        datasets = ["iris", "adult", "digits"]
        data_path = joinpath(dirname(pathof(DecisionTree)), "..", "test/data/")

        if name == "digits"
            f = open(joinpath(data_path, "digits.csv"))
            data = readlines(f)[2:end]
            data = [[parse(Float32, i)
                for i in split(row, ",")]
                for row in data]
            data = hcat(data...)
            Y = Int.(data[1, 1:end]) .+ 1
            X = convert(Matrix, transpose(data[2:end, 1:end]))
            return X, Y
        end

        if name == "iris"
            iris = DelimitedFiles.readdlm(joinpath(data_path, "iris.csv"), ',')
            X = iris[:, 1:4]
            Y = iris[:, 5]
            return X, Y
        end

        if name == "adult"
            adult = DelimitedFiles.readdlm(joinpath(data_path, "adult.csv"), ',');
            X = adult[:, 1:14];
            Y = adult[:, 15];
            return X, Y
        end

        if !(name in datasets)
            throw("Available datasets are $(join(datasets,", "))")
        end
    end

    mk_rng(rng::Random.AbstractRNG) = rng
    mk_rng(seed::T) where T <: Integer = Random.MersenneTwister(seed)

    # convert between two tree representations
    function light_regressor(tree, labels, method) where {S, T}

        function recurse(node, list)
            if node.is_leaf
                return Leaf{T}(node.label, node.deviation, node.region)
            else
                l = lighten(node.l, list)
                r = lighten(node.r, list)
                return Node{S, T}(node.feature, node.threshold, l, r)
            end
        end

        function main(tree, labels)
            root = recurse(tree.root, tree.list)
            if labels == nothing
                return Tree(root, nothing, method)
            else
                labs = labels[tree.labels]
                return Tree(root, labs, method)
            end
        end

    end

    # convert between two tree representations
    function light_classifier(tree, labels, method) where {S, T}

        function recurse(node, list)
            if node.is_leaf
                label = list[node.label]
                return Leaf{T}(label, node.deviation, node.region)
            else
                l = lighten(node.l, list, labels)
                r = lighten(node.r, list, labels)
                return Node{S, T}(node.feature, node.threshold, l, r)
            end
        end

        function main(tree, labels)
            root = recurse(tree.root, tree.list)
            if labels == nothing
                return Tree(root, nothing, method)
            else
                labs = labels[tree.labels]
                return Tree(root, labs, method)
            end
        end
    end

    function prune_tree(tree::LeafOrNode{S, T}, purity_thresh=1.0) where {S, T}

        function recursive_assign(leaf::Leaf{T}, set::Set{T})
            for item in leaf.values
                push!(set, item)
            end
        end

        function recursive_assign(node::Node{S, T}, set::Set{T})
            recursive_assign(node.left, set)
            recursive_assign(node.right, set)
        end

        function recurse(
                leaf          :: Leaf{T},
                purity_thresh :: Float64,
                label2int     :: Dict{T, Int},
                labels        :: Vector{T})
            nc = fill(0.0, length(labels))
            for i in leaf.values
                nc[label2int[i]] += 1.0
            end
            return nc, leaf
        end

        function recurse(
                node          :: Node{S, T},
                purity_thresh :: Float64,
                label2int     :: Dict{T, Int},
                labels        :: Vector{T})

            ncl, l = recurse(node.left, purity_thresh, label2int, labels)
            ncr, r = recurse(node.right, purity_thresh, label2int, labels)

            if is_leaf(l) && is_leaf(r)

                @simd for i in 1:length(labels)
                    ncl[i] += ncr[i]
                end

                n_samples = length(l.values) + length(r.values)
                purity = -util.entropy(ncl, n_samples)
                if purity > purity_thresh
                    return ncl, Leaf{T}(labels[argmax(ncl)], [l.values; r.values])
                end

            end

            return ncl, Node{S, T}(node.featid, node.featval, l, r)
        end

        function main(tree::LeafOrNode{S, T}, purity_thresh=1.0)
            set = Set{T}()
            recursive_assign(tree, set)
            labels = collect(set)
            label2int  = Dict{T, Int}(label=>i for (i, label) in enumerate(labels))

            ncl, node = recurse(tree, purity_thresh, label2int, labels)

            return node
        end

        return main(tree, 1.0-purity_thresh)
    end

end



module Apply

    import ..Struct: Ensemble, Tree, Leaf, Node, is_leaf

    function take_label(x::Leaf{T})::T where {T}
        return x.label
    end

    function mean(out)
        return sum(out) / length(out)
    end

    function majority(out)
        counts = {}
        for i in out
            counts[i] = get(counts, i, 0) + 1
        end
        return findmax(counts)[1]
    end

    function weighted_majority(out, w)
        counts = {}
        for i in 1:length(out)
            item = out[i]
            counts[item] = get(counts, item, 0) + w[i]
        end

        return findmax(counts)[1]
    end


    #= APPLY: TREE =#
    function _apply(
            tree        :: Tree{S, T},
            features    :: Matrix{S},
            mapper      :: Function
            index       :: Int) where {S, T}
        curr = tree.root
        while !Struct.is_leaf(curr)
            if features[index, tree.featid] < tree.featval
                curr = curr.left
            else
                curr = curr.right
            end
        end

        return mapper(curr)
    end

    function apply(
            tree        :: Tree{S, T},
            features    :: Vector{S}
            mapper      :: Function = take_label) where {S, T}
        curr = tree.root
        while !Struct.is_leaf(curr)
            if features[tree.featid] < tree.featval
                curr = curr.left
            else
                curr = curr.right
            end
        end

        return mapper(curr)
    end

    # works for any array dimension
    function apply(
            tree        :: Tree{S, T},
            features    :: Array{S}
            mapper      :: Function = take_label) where {S, T}

        dims = shape(features)
        if size(features) == 0
            return reshape([], dims[1:end-1])
        end

        features    = reshape(features, :, dims[end])
        n_samples   = size(features, 1)
        # take the first index to get the return type
        tester      = _apply(tree, features, mapper, 1)
        predictions = Array{typeof(tester)}(undef, n_samples)
        predictions[1] = tester
        for ind in 2:n_samples
            predictions[i] = _apply(tree, features, mapper, ind)
        end

        return reshape(predictions, dims[1:end-1])
    end


    #= APPLY: ENSEMBLE =#

    function _apply(
            ensemble    :: Ensemble{S, T},
            features    :: Vector{S},
            reducer     :: Function,
            mapper      :: Function,
            ind         :: Int) where {S, T}
        trees       = ensemble.trees
        tester      = apply(trees[1], features, mapper, ind)
        predictions = Array{typeof(tester)}(undef, n_samples)
        predictions[1] = tester
        for i in 2:trees.length
            predictions[i] = _apply(trees[i], features, mapper, ind)
        end

        return reducer(predictions)
    end

    function apply(
            ensemble    :: Ensemble{S, T},
            features    :: Vector{S},
            reducer     :: Function,
            mapper      :: Function = take_label) where {S, T}
        if length(ensemble.trees) == 0
            throw ("empty ensemble")
        end
        trees       = ensemble.trees
        tester      = apply(trees[1], features, mapper)
        predictions = Array{typeof(tester)}(undef, n_samples)
        predictions[1] = tester
        for i in 2:trees.length
            predictions[i] = apply(trees[i], features, mapper)
        end

        return reducer(predictions)
    end

    # works for any array dimension
    function apply(
            ensemble    :: Ensemble{S, T},
            features    :: Array{S},
            reducer     :: Function,
            mapper      :: Function = take_label) where {S, T}
        if length(ensemble.trees) == 0
            throw ("empty ensemble")
        end
        dims = shape(features)
        if size(features) == 0
            return reshape([], dims[1:end-1])
        end

        trees       = ensemble.trees
        features    = reshape(features, :, dims[end])
        n_samples   = size(features, 1)

        tester      = _apply(ensemble, features, reducer, mapper, 1)
        predictions = Array{typeof(tester)}(undef, n_samples)
        predictions[1] = tester

        for ind in 2:n_samples
            predictions[i] = _apply(ensemble, features, reducer, mapper, ind)
        end

        return reshape(predictions, dims[1:end-1])
    end

    function apply(ensemble::Ensemble{S, T}, features::Array{S})
        if ensemble.method == "forest" && T <: Float64
            return apply(ensemble, features, mean, take_label)
        elseif ensemble.method == "forest"
            return apply(ensemble, features, majority, take_label)
        else
            throw("ensemble type unknown: please pass "
                * "explicit reducer and mapper as arguments.")
        end
    end
end