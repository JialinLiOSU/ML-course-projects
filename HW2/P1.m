% HW2 of Machine Learning Class Problem 1
train = load('train79.mat');
test = load('test79.mat');
% the number of observated data in training dataset
n = length(train.d79);
y1=7*ones(1000,1);
y2=9*ones(1000,1);
Y = [y1;y2];
%% SVM Classifier
SVM_Md1 = fitclinear(train.d79,Y);
% test the SVM classifier
[SVM_label,SVM_score] = predict(SVM_Md1,test.d79);
% prediction accuracy rate
diff=abs(SVM_label-Y)/2;
SVM_err = (sum(diff))/2000
%% least squares linear classifier
Y_ls=[ones(1000,1);-1*ones(1000,1)];
X_ls=[train.d79,ones(n,1)];
W=lsqlin(X_ls,Y_ls);
X_ls_test=[test.d79,ones(n,1)];
LS_label=sign(X_ls_test*W);
LS_err = (1/2*(sum(abs(Y_ls-LS_label))))/2000

%%  changing Penalty term
Lambda = logspace(-10,-1,10);
SVM_Md1 = fitclinear(train.d79,Y,'Regularization','ridge','lambda',Lambda);
[SVM_label,SVM_score] = predict(SVM_Md1,test.d79);
SVM_MisC_list=zeros(length(Lambda),1);
% prediction accuracy rate
for i=1:length(Lambda)
    diff=abs(SVM_label(:,i)-Y)/2;
    SVM_MisC_list(i)= sum(diff)/2000;
end
plot(SVM_MisC_list,'r')
title('Error rate of linear svm with different C value')
xticks([1:1:10])
xticklabels({'1e-10','1e-9','1e-8','1e-7','1e-6','1e-5','1e-4','1e-3','1e-2','1e-1'})
xlabel('c value');
ylabel('Error rate');
