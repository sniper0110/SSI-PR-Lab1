function [theta] = linearReg(X, y)

theta = zeros(size(X, 2), 1);

theta = X\y;

end
