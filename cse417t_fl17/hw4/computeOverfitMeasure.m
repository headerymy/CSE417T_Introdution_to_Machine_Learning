function [ overfit_m ] = computeOverfitMeasure( true_Q_f, N_train, N_test, var, num_expts )
%COMPUTEOVERFITMEASURE Compute how much worse H_10 is compared with H_2 in
%terms of test error. Negative number means it's better.
%   Inputs
%       true_Q_f: order of the true hypothesis
%       N_train: number of training examples
%       N_test: number of test examples
%       var: variance of the stochastic noise
%       num_expts: number of times to run the experiment
%   Output
%       overfit_m: vector of length num_expts, reporting each of the
%                  differences in error between H_10 and H_2
for i = 1:num_expts
[train_set, test_set] = generate_dataset( true_Q_f, N_train, N_test, var^0.5);
z2 = computeLegPoly(train_set(:,1),2);
z10 = computeLegPoly(train_set(:,1),10);

w2 = glmfit(z2, train_set(:,2), 'normal', 'constant', 'off');
w10 = glmfit(z10, train_set(:,2), 'normal', 'constant', 'off');

g2 = z2*b2;
g10 = z10*b10;

E2 = mean((g2(:,1)-test_set(:,2)).^2);
E10 = mean((g10(:,1)-test_set(:,2)).^2);
overfit_m(i) = E10 - E2;
end
end
