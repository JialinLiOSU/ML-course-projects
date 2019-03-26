% HW2 of Machine Learning Class Problem 2
train = load('train79.mat');
test = load('test79.mat');
% the number of observated data in training dataset
[n,d]= size(train.d79);

%% least squares linear classifier implemented by myself
Y_ls=[ones(1000,1);-1*ones(1000,1)];
X_ls=[ones(n,1),train.d79];
% theta=zeros(d+1,1);
% iterations = 1500;
% alpha = 1*10e-11;
% computeCost(X_ls, Y_ls, theta);
% theta = gradientDescent(X_ls, Y_ls, theta, alpha, iterations);
% X_ls_test=[ones(n,1),test.d79];
% LS_label=sign(X_ls_test*theta);
% 
% LS_err=1/2*(sum(abs(Y_ls-LS_label)))/2000
% LS_Acc = (2000-1/2*(sum(abs(Y_ls-LS_label))))/2000;

Y_ls=[ones(1000,1);-1*ones(1000,1)];
W=lsqlin(X_ls,Y_ls);
X_ls_test=[ones(n,1),test.d79];
LS_label=sign(X_ls_test*W);
LS_err_stand = 1/2*(sum(abs(Y_ls-LS_label)))/2000