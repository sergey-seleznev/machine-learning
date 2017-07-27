function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

m = length(y);

h = (X * theta);
J = 1 / (2 * m) * sum((h - y) .^ 2);

reg_theta = theta;
reg_theta(1) = 0;
J = J + lambda / (2 * m) * sum(reg_theta .^ 2);

grad = 1 / m * (X' * (h - y)) + lambda / m * reg_theta;
grad = grad(:);

end
