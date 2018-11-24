
module Struct

    import Base: length, show

    struct Leaf{T}
        label      :: T

        # one summary statistic for fast retrieval
        # - probability of inaccuracy for classication
        # - variance of leaf for regression
        # deviation  :: Float64

        # the location in the tree.labels
        # where the labels in this leaf are stored
        region     :: UnitRange{Int}
    end

    struct Node{S, T}
        featid  :: Int
        featval :: S
        left    :: Union{Leaf{T}, Node{S, T}}
        right   :: Union{Leaf{T}, Node{S, T}}
    end

    struct Tree{S, T}
        root    :: Union{Leaf{T}, Node{S, T}}
        # nothing if we choose not to store any thing
        # otherwise, a vector of labels
        labels  :: Union{Nothing, Vector{T}}
        # how the tree was constructed, criterion, etc.
        method  :: String
    end

    struct Ensemble{S, T}
        trees  :: Vector{Tree{S, T}}
        coeffs :: Union{Nothing, Vector{Float64}}

        # how the ensemble was constructed
        method :: String
    end

    is_leaf(l::Leaf) = true
    is_leaf(n::Node) = false

    length(leaf::Leaf) = 1
    length(tree::Node) = length(tree.left) + length(tree.right)
    length(tree::Tree) = length(tree.root)
    length(ensemble::Ensemble) = length(ensemble.trees)

    depth(leaf::Leaf) = 0
    depth(tree::Node) = 1 + max(depth(tree.left), depth(tree.right))
    depth(tree::Tree) = depth(tree.root)

    function print_tree(leaf::Leaf, depth=-1, indent=0)
        n_samples = leaf.region.stop - leaf.region.start + 1
        println("$(leaf.label) : $(leaf.deviation) : $(n_samples) samples")
    end

    function print_tree(node::Node, depth=-1, indent=0)
        if depth == indent
            println()
            return
        end
        println("Feature $(node.featid), Threshold $(node.featval)")
        print("    " ^ indent * "L-> ")
        print_tree(node.left, depth, indent + 1)
        print("    " ^ indent * "R-> ")
        print_tree(node.right, depth, indent + 1)
    end

    function print_tree(tree::Tree)
        print_tree(tree.root)
    end

    function show(io::IO, leaf::Leaf)
        n_samples = leaf.region.stop - leaf.region.start + 1
        println(io, "Decision Leaf")
        println(io, "Label     : $(leaf.labels)")
        println(io, "Deviation : $(leaf.deviation)")
        print(io,   "Samples   : $(n_samples)")
    end

    function show(io::IO, node::Node)
        println(io, "Decision Node")
        println(io, "Leaves : $(length(node))")
        print(io,   "Depth  : $(depth(node))")
    end

    function show(io::IO, tree::Tree)
        println(io, "Decision Tree")
        println(io, "Type   : $(tree.method)")
        println(io, "Leaves : $(length(tree))")
        print(io,   "Depth  : $(depth(tree))")
    end

    function show(io::IO, ensemble::Ensemble)
        println(io, "Ensemble of Decision Trees")
        println(io, "Trees:      $(length(ensemble))")
        println(io, "Avg Leaves: $(mean([length(tree) for tree in ensemble.trees]))")
        print(io,   "Avg Depth:  $(mean([depth(tree) for tree in ensemble.trees]))")
    end

end # module


module Misc

    import Random
    import DelimitedFiles
    import ..Struct: Tree, Leaf, Node, is_leaf

    mk_rng(rng::Random.AbstractRNG) = rng
    mk_rng(seed::T) where T <: Integer = Random.MersenneTwister(seed)

    function load_data(name)
        datasets = ["iris", "adult", "digits"]
        # data_path = joinpath(dirname(pathof(DecisionTree)), "..", "test/data/")
        data_path = joinpath(".", "test/data/")

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

    function prune_tree!(tree::Tree{S, T}, purity_thresh=1.0) where {S, T}

        function recursive_assign(leaf::Leaf{T}, set::Set{T})
            @simd for item in leaf.values
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
            @simd for i in leaf.values
                @inbounds nc[label2int[i]] += 1.0
            end
            return nc, leaf
        end

        function recurse!(
                node          :: Node{S, T},
                purity_thresh :: Float64,
                label2int     :: Dict{T, Int},
                labels        :: Vector{T})

            ncl, l = recurse(node.left, purity_thresh, label2int, labels)
            ncr, r = recurse(node.right, purity_thresh, label2int, labels)

            if is_leaf(l) && is_leaf(r)

                @simd for i in 1:length(labels)
                    @inbounds ncl[i] += ncr[i]
                end

                n_samples = length(l.values) + length(r.values)
                purity = -util.entropy(ncl, n_samples)
                if purity > purity_thresh
                    return ncl, Leaf{T}(labels[argmax(ncl)], [l.values; r.values])
                end

            end

            return ncl, Node{S, T}(node.featid, node.featval, l, r)
        end

        function main(tree::Tree{S, T}, purity_thresh=1.0)
            set = Set{T}()
            recursive_assign(tree.root, set)
            labels = collect(set)
            label2int  = Dict{T, Int}(label=>i for (i, label) in enumerate(labels))

            ncl, node = recurse(tree.root, purity_thresh, label2int, labels)

            return tree
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
        counts = Dict()
        for i in out
            counts[i] = get(counts, i, 0) + 1
        end
        return findmax(counts)[2]
    end

    function weighted_majority(out, w)
        counts = Dict()
        for i in 1:length(out)
            item = out[i]
            counts[item] = get(counts, item, 0) + w[i]
        end

        return findmax(counts)[2]
    end


    #= APPLY: TREE =#
    function _apply(
            tree        :: Tree{S, T},
            features    :: Matrix{S},
            mapper      :: Function,
            index       :: Int) where {S, T}
        curr = tree.root
        while !is_leaf(curr)
            if features[index, curr.featid] < curr.featval
                curr = curr.left
            else
                curr = curr.right
            end
        end

        return mapper(curr)
    end

    function apply(
            tree        :: Tree{S, T},
            features    :: Vector{S},
            mapper      :: Function = take_label) where {S, T}
        curr = tree.root
        while !is_leaf(curr)
            if features[curr.featid] < curr.featval
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
            features    :: Array{S},
            mapper      :: Function = take_label) where {S, T}

        dims = size(features)
        if size(features) == 0
            return reshape([], dims[1:end-1])
        end

        # TODO : @inbounds this
        features    = reshape(features, :, dims[end])
        n_samples   = size(features, 1)
        # take the first index to get the return type
        tester      = _apply(tree, features, mapper, 1)
        predictions = Array{typeof(tester)}(undef, n_samples)
        @inbounds predictions[1] = tester
        # @inbounds @simd 
        for i in 2:n_samples
            predictions[i] = _apply(tree, features, mapper, i)
        end

        return reshape(predictions, dims[1:end-1])
    end


    #= APPLY: ENSEMBLE =#

    function _apply(
            ensemble    :: Ensemble{S, T},
            features    :: Matrix{S},
            mapper      :: Function,
            ind         :: Int) where {S, T}
        n_samples   = size(features, 1)
        trees       = ensemble.trees
        tester      = _apply(trees[1], features, mapper, ind)
        filler      = Array{typeof(tester)}(undef, length(trees))
        @inbounds filler[1] = tester
        @inbounds @simd for i in 2:length(trees)
            filler[i] = _apply(trees[i], features, mapper, ind)
        end

        return filler
    end

    function _apply(
            ensemble    :: Ensemble{S, T},
            features    :: Matrix{S},
            reducer     :: Function,
            mapper      :: Function,
            filler      :: Vector{U},
            ind         :: Int) where {S, T, U}
        trees       = ensemble.trees
        n_outputs   = length(trees)
        @assert n_outputs == length(filler)
        @inbounds @simd for i in 1:length(trees)
            filler[i] = _apply(trees[i], features, mapper, ind)
        end

        return reducer(filler)
    end


    function apply(
            ensemble    :: Ensemble{S, T},
            features    :: Vector{S},
            reducer     :: Function,
            mapper      :: Function = take_label) where {S, T}
        if length(ensemble.trees) == 0
            throw("empty ensemble")
        end
        trees  = ensemble.trees
        tester = apply(trees[1], features, mapper)
        filler = Array{typeof(tester)}(undef, length(trees))
        @inbounds predictions[1] = tester
        @inbounds @simd for i in 2:length(trees)
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
            throw("empty ensemble")
        end
        dims = size(features)
        if size(features) == 0
            return reshape([], dims[1:end-1])
        end

        trees       = ensemble.trees
        features    = reshape(features, :, dims[end]) # TODO : @inbounds this
        n_samples   = size(features, 1)

        treeoutputs = _apply(ensemble, features, mapper, 1)
        testoutput  = reducer(treeoutputs)
        predictions = Array{typeof(testoutput)}(undef, n_samples)
        @inbounds predictions[1] = testoutput
        # @inbounds @simd 
        for ind in 2:n_samples
            predictions[ind] = _apply(ensemble, features, reducer, 
                                      mapper, treeoutputs, ind)
        end

        return reshape(predictions, dims[1:end-1])
    end

    function apply(ensemble::Ensemble{S, T}, features::Array{S}) where {S, T}
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

