module Measures

    using LinearAlgebra
    
    struct ConfusionMatrix
        classes  :: Vector
        matrix   :: Matrix{Int}
        accuracy :: Float64
        kappa    :: Float64 # cohen's kappa
    end

    function confusion_matrix(actual::Vector, predicted::Vector)

        function assign(a, b)
            @assert length(a) == length(b)
            # TODO: make this more memory efficient
            list = sorted(unique([a; b]))

            dict = Dict{T, Int}()
            @simd for i in 1:length(list)
                @inbounds dict[list[i]] = i
            end

            _a = Array{Int}(undef, length(Y))
            _b = Array{Int}(undef, length(Y))
            @simd for i in 1:length(a)
                @inbounds _a[i] = dict[a[i]]
                @inbounds _b[i] = dict[b[i]]
            end

            return list, _a, _b
        end

        function main(actual, predicted)
        end

        @assert length(actual) == length(predicted)

        classes, actual, predicted = assign(actual, predicted)
        n_classes = length(classes)
        n_samples = length(actual)
        conf_matr = zeros(Int, n_classes, n_classes)
        for (a, p) in zip(actual, predicted)
            conf_mat[a, p] += 1
        end

        accuracy = LinearAlgebra.tr(conf_mat) / n_samples
        prob_chance = (sum(CM,dims=1) * sum(CM,dims=2))[1] / sum(CM)^2
        kappa = (accuracy - prob_chance) / (1.0 - prob_chance)
        return ConfusionMatrix(classes, confusion, accuracy, kappa)
    end


end