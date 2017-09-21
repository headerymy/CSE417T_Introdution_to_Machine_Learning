function [num_iters, bounds] =  perceptron_experiment(N, d, num_samples)
%perceptron_experiment Code for running the perceptron experiment in HW1
%   Inputs: N is the number of training examples
%           d is the dimensionality of each example (before adding the 1)
%           num_samples is the number of times to repeat the experiment
%   Outputs: num_iters is the # of iterations PLA takes for each sample
%            bound_minus_ni is the difference between the theoretical bound
%               and the actual number of iterations
%      (both the outputs should be num_samples long)
    
    for n = 1 : num_samples
        % set the weight vector
        weight = [0 rand(1, d)]';
        % random traning set
        training = 2 * rand(d, N) - 1;
        
        % get the label
        label = sign(weight(2:end,:)' * training);
        data = [training; label]';
        
        % get weight and iteration
        [weight_learn, iteration] = perceptron_learn(data);
        num_iters(n) = iteration;
        
        diff = sum(abs(weight_learn - weight(2:end,:)'));
        
        max_R = max(norm(training));
        min2_P = min(label .* ((weight(2:end,:)') * training))^2;
        bounds(n) = (max_R * norm(weight))/min2_P;
        diff(n) = bounds(n) - num_iters(n);
    end
    
    figure(1)
    hist(num_iters)
    title('Histogram: # of iterations')
    
    figure(2)
    hist(log10(bounds))
    title('Histogram: difference between # of iterations and bound') 
    
end