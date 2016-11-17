function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

params = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
result_calc = zeros(64,3);
indx = 0;

for C_indx = 1:size(params,2) 
    for sigma_index = 1:size(params,2)
        indx = indx + 1;
        C_test = params(C_indx);
        sigma_test = params(sigma_index);
        model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
        predictions = svmPredict(model, Xval);
        err = mean(double(predictions ~= yval));
        result_calc(indx,:) = [err, C_test, sigma_test];
    end
end

indx = find(result_calc(:,1) == min(result_calc(:,1)));

C = result_calc(indx,2);
sigma = result_calc(indx,3);


% =========================================================================

end
