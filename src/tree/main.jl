
module BuildTree
	include("tree/util.jl")
	include("tree/classifier.jl")
	include("tree/regressor.jl")

	import ..Struct: Tree, Leaf, Node
	import ..BuildTree
    # convert between two tree representations
    function light_regressor(
    		tree 	:: BuildTree.Regressor.Tree{S},
    		labels	:: Union{Nothing, Vector{T}},
    		method	:: String) where {S, T <: Float64}

        function recurse(node)
            if node.is_leaf
                return Leaf{Float64}(node.label, node.region)
            else
                l = recurse(node.l)
                r = recurse(node.r)
                return Node{S, Float64}(node.feature, node.threshold, l, r)
            end
        end

        function main(tree, method)
            root = recurse(tree.root)
            if labels == nothing
                return Tree(root, nothing, method)
            else
                labs = labels[tree.labels]
                return Tree(root, labs, method)
            end
        end

        return main(tree, method)
    end

    # convert between two tree representations
    function light_classifier(
    		tree 	:: BuildTree.Classifier.Tree{S, T},
    		labels	:: Union{Nothing, Vector{T}},
    		method	:: String) where {S, T}

        function recurse(node, list)
            if node.is_leaf
                label = list[node.label]
                return Leaf{T}(label, node.region)
            else
                l = recurse(node.l, list)
                r = recurse(node.r, list)
                return Node{S, T}(node.feature, node.threshold, l, r)
            end
        end

        function main(tree, labels, method)
            root = recurse(tree.root, tree.list)
            if labels == nothing
                return Tree(root, nothing, method)
            else
                labs = labels[tree.labels]
                return Tree(root, labs, method)
            end
        end

        return main(tree, labels, method)
    end

end

# include("classification/main.jl")
# include("regression/main.jl")