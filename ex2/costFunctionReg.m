function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y);
h = sigmoid(theta' * X')';

reg_theta = theta;
reg_theta(1) = 0;

J = (1 / m) * sum(-y' * log(h) - (1 - y') * log(1 - h)) ...
    + lambda / (2 * m) * sum(reg_theta .^ 2);

grad = ((1 / m) * (h - y)' * X)' + (lambda / m * reg_theta);

end
