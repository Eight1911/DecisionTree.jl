
module Struct

    import Base: length, show

    struct Leaf{T}
        label      :: T
        # one or two summary statistics for fast retrieval
        # - probability of inaccuracy for classication
        # - variance of leaf for regression
        deviation  :: Float64
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
        # nothing if we choose not to store
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

    function print_tree(tree::Node, depth=-1, indent=0)
        if depth == indent
            println()
            return
        end
        println("Feature $(tree.featid), Threshold $(tree.featval)")
        print("    " ^ indent * "L-> ")
        print_tree(tree.left, depth, indent + 1)
        print("    " ^ indent * "R-> ")
        print_tree(tree.right, depth, indent + 1)
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