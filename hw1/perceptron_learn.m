function [w, iterations] = perceptron_learn(data_in)
%perceptron_learn Run PLA on the input data
%   Inputs: data_in: Assumed to be a matrix with each row representing an
%                    (x,y) pair, with the x vector augmented with an
%                    initial 1, and the label (y) in the last column
%   Outputs: w: A weight vector (should linearly separate the data if it is
%               linearly separable)
%            iterations: The number of iterations the algorithm ran for
   
    x = data_in( : , 1 : (end - 1));
    label = data_in( : ,end);
    num_iters = 0;
    
    [num, dim] = size(x);    
    weight_learn = zeros([dim 1])';
    condition = 0;
    
    while not(condition)
        condition = 1;
        for n = 1 : num
            label_learn = sign(weight_learn * x');
            if label_learn(n) ~= label(n)
                weight = weight_learn + x(n, :)*label(n);
                condition = 0;
            end
        end
        weight_learn = weight;
        num_iters = num_iters + 1;
    end
    w = weight_learn;
    iterations = num_iters;
end

