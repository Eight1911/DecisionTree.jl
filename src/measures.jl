module Measures

    import Base: show
    using LinearAlgebra

    
    struct ConfusionMatrix
        classes  :: Vector
        matrix   :: Matrix{Int}
        accuracy :: Float64
        kappa    :: Float64 # cohen's kappa
    end

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
        end

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