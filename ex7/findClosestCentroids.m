function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

m = size(X, 1);
o = ones(K, 1);

idx = zeros(m, 1);

for i = 1:m
  x = X(i, :);
  lengths = sqrt(sum(((o * x) - centroids) .^ 2, 2));
  [~, ind] = min(lengths);
  idx(i) = ind;
end

end
