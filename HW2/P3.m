% HW2 of Machine Learning Class Problem 3
train = load('train79.mat');
test = load('test79.mat');
% the number of observated data in training dataset
n = length(train.d79);
%% PCA reduce the dimension

y1=7*ones(1000,1);
y2=9*ones(1000,1);
Y = [y1;y2];
%% SVM Classifier
SVM_Md1 = fitclinear(train.d79,Y);
% test the SVM classifier
[SVM_label,SVM_score] = predict(SVM_Md1,test.d79);
% prediction accuracy rate
diff=abs(SVM_label-Y)/2;
SVM_Acc = (2000 - sum(diff))/2000
%% least squares linear classifier
Y_ls=[ones(1000,1);-1*ones(1000,1)];
X_ls=[train.d79,ones(n,1)];
W=lsqlin(X_ls,Y_ls);
LS_label=sign(X_ls*W);
LS_Acc = (2000-1/2*(sum(abs(Y_ls-LS_label))))/2000