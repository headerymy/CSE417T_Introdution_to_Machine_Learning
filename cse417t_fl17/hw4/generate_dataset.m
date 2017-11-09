function [ train_set test_set ] = generate_dataset( Q_f, N_train, N_test, sigma )
%GENERATE_DATASET Generate training and test sets for the Legendre
%polynomials example
%   Inputs:
%       Q_f: order of the hypothesis
%       N_train: number of training examples
%       N_test: number of test examples
%       sigma: standard deviation of the stochastic noise
%   Outputs:
%       train_set and test_set are both 2-column matrices in which each row
%       represents an (x,y) pair
train_data = 2*rand(N_train,1)-1;
test_data = 2*rand(N_test,1)-1;
train = normrnd(0,1,N_train,1);
test = normrnd(0,1,N_test,1);
aq = normrnd(0,1,Q_f+1,1);

c = 0;
for d = 0:Q_f
c = c + 1/(2*d+1);
end
train_set(:,2) = c^0.5 * computeLegPoly(train_data(:,1), Q_f) * aq + train(:,1) * sigma;
test_set(:,2) = c^0.5 * computeLegPoly(test_data(:,1), Q_f) * aq + test(:,1) * sigma;
end

