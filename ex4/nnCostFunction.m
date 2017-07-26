function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);

% Forward Propagation

a1 = X;
a1 = [ones(m, 1) a1];

z2 = (Theta1 * a1')';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];

z3 = (Theta2 * a2')';
a3 = sigmoid(z3);

[maxs, h] = max(a3, [], 2);

% Cost Calculation
J = 0;
for l = 1:num_labels
  yl = (y == l);
  hl = a3(:, l);
  J = J + (1 / m) * sum(-yl' * log(hl) - (1 - yl') * log(1 - hl));
end

% Cost Regularization
J = J + lambda / (2 * m) * ( ...
  sum(sum(Theta1(:, 2:size(Theta1, 2)) .^ 2)) + ...
  sum(sum(Theta2(:, 2:size(Theta2, 2)) .^ 2))
);

% Back Propagation
D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));

for t = 1:m

  a1 = X(t, :)';
  a1 = [1; a1];

  z2 = Theta1 * a1;
  a2 = sigmoid(z2);
  a2 = [1; a2];

  z3 = Theta2 * a2;
  a3 = sigmoid(z3);

  yk = [1:num_labels]' == y(t);

  d3 = a3 - yk;
  
  d2 = (Theta2' * d3) .* sigmoidGradient([0; z2]);
  d2 = d2(2:end);

  D1 = D1 + d2 * a1';
  D2 = D2 + d3 * a2';

end

% Gradient Regularization
reg1 = lambda / m * Theta1;
reg1(:, 1) = zeros(size(reg1, 1), 1);
Theta1_grad = 1 / m * D1 + reg1;

reg2 = lambda / m * Theta2;
reg2(:, 1) = zeros(size(reg2, 1), 1);
Theta2_grad = 1 / m * D2 + reg2;

% Gradient Unrolling
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
