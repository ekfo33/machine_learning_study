function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
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

C_p = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_p = [0.01 0.03 0.1 0.3 1 3 10 30];

C_length = length(C_p);
sigma_length = length(sigma_p);

err_predict = inf;

for i = 1:C_length
    for j = 1:sigma_length        
        model = svmTrain(X, y, C_p(1, i), @(x1, x2) gaussianKernel(x1, x2, sigma_p(1, j)));        
        err = mean(double(svmPredict(model, Xval) ~= yval));        
        fprintf('C = %f, sigma = %f, err = %f\n', C_p(1, i), sigma_p(1, j), err);
        if (err <= err_predict) 
            C =  C_p(1, i);
            sigma = sigma_p(1, j);
            err_predict = err;
        end    
    end    
end

% =========================================================================

end
