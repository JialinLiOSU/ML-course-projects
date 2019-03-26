function J = computeCost(X, Y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(Y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

J = (transpose(theta)*transpose(X)*X*theta-2*theta*transpose(theta)*transpose(X)*Y+transpose(Y)*Y)/m;
% =========================================================================

end
