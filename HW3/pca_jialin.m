function [U, S] = pca_jialin(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

sigma = X'*X/m;% Compute covariance matrix
[U, S, V] = svd(sigma);% Compute eigen vectors

end