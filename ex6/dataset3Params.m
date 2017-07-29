function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

C = 1;
sigma = 0.1;

% DETERMINING OPTIMAL VALUES:
% options = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30 ];
% optionCount = length(options);
% minErr = 10^9;
% for i = 1:optionCount
%   currentC = options(i);
%   for j = 1:optionCount
%     currentSigma = options(j);
%     model = svmTrain(X, y, currentC, @(x1, x2) gaussianKernel(x1, x2, currentSigma));
%     predictions = svmPredict(model, Xval);
%     err = mean(double(predictions ~= yval));
%     if (err < minErr)
%       minErr = err;
%       C = currentC;
%       sigma = currentSigma;
%     end
%   end
% end

end
