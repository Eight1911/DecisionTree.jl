include("../src/DecisionTree.jl")
using Main.DecisionTree

include("test_suites.jl")



#= temporarily commenting out new prune_tree implementation
function prune_tree(tree::LeafOrNode{S, T}, purity_thresh=0.0) where {S, T}

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

    return main(tree, purity_thresh)
end
=#




#= temporarily commenting out new prune_tree implementation
function prune_tree(tree::LeafOrNode{S, T}, purity_thresh=0.0) where {S, T <: Float64}

    function recurse(leaf :: Leaf{T}, purity_thresh :: Float64)
        tssq = 0.0
        tsum = 0.0
        for v in leaf.values
            tssq += v*v
            tsum += v
        end

        return tssq, tsum, leaf
    end

    function recurse(node :: Node{S, T}, purity_thresh :: Float64)

        lssq, lsum, l = recurse(node.left, purity_thresh)
        rssq, rsum, r = recurse(node.right, purity_thresh)

        if is_leaf(l) && is_leaf(r)
            n_samples = length(l.values) + length(r.values)
            tsum = lsum + rsum
            tssq = lssq + rssq
            tavg = tsum / n_samples
            purity = tavg * tavg - tssq / n_samples
            if purity > purity_thresh
                return tsum, tssq, Leaf{T}(tavg, [l.values; r.values])
            end

        end

        return 0.0, 0.0, Node{S, T}(node.featid, node.featval, l, r)
    end

    _, _, node = recurse(tree, purity_thresh)
    return node
end
=#
