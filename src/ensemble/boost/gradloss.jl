
module GradientLossFunction

    # lol, doesn't actually have the loss function
    struct Loss
        # minus_gradient : Float64 x Float64 -> Float64
        # returns the negative gradient of the loss
        # with respect to the prediction
        # i.e. returns dL(arg1, arg2)  / d arg2
        minus_gradient :: Function
        # point_minimize : 
        #   : List{Float64} x List{Float64} x UnitRange  -> Float64
        # returns the minimizer of (arg1 - arg2)[unitrange]
        # e.g, returns the mean for L2 loss and median for L1 loss
        point_minimize :: Function
    end

    # L2 Loss for Regression
    function proto_L2()

        function minus_gradient(y, p)
            return p - y
        end

        function point_minimize(R, space, region)
            tsum = 0.0
            for i in region
                tsum += R[i]
            end
            return tsum / length(region)
        end

        return Loss(minus_gradient, point_minimize)
    end

    function proto_L1()

        function minus_gradient(y, p)
            return if p > y
                1
            elseif p < y
                -1
            else
                0
            end
        end

        function point_minimize(R, space, region)
            # space is just stack space, cuz its faster
            # return the median of (P - Y)[region]
            return median(R[region])
        end
    end

    l1 = proto_L1()
    l2 = proto_L2()

end
