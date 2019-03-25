% HW2 of Machine Learning Class Problem 2
train = load('train79.mat');
test = load('test79.mat');
% the number of observated data in training dataset
n = length(train.d79);

%% least squares linear classifier implemented by myself
Y_ls=[ones(1000,1);-1*ones(1000,1)];
X_ls=[train.d79,ones(n,1)];


W=lsqlin(X_ls,Y_ls);
LS_label=sign(X_ls*W);
LS_Acc = (2000-1/2*(sum(abs(Y_ls-LS_label))))/2000;