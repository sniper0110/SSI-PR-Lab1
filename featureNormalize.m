function [Xn, mu, sigma] = featureNormalize(X)

Xn = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

mu = mean(X);
sigma = std(X);

for i=1:size(X,1)
    X(i,:) = (X(i,:) - mu) ./ sigma;
end

Xn = X;

end
