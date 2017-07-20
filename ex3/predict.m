function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

num_labels = size(Theta2, 1);

a1 = [ones(size(X, 1), 1) X];

z2 = (Theta1 * a1')';
a2 = sigmoid(z2);
a2 = [ones(size(a2, 1), 1) a2];

z3 = (Theta2 * a2')';
a3 = sigmoid(z3);

[m, p] = max(a3, [], 2);

end
