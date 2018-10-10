
module Tree
	include("tree/util.jl")
	include("tree/classifier.jl")
	include("tree/regressor.jl")
end

include("classification/main.jl")
include("regression/main.jl")