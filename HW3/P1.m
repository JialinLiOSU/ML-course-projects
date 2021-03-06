% HW3 of Machine Learning Class Problem 1 about DT and Ensemble Learning
clear
train = load('train79.mat');
X_train=train.d79;
test = load('test79.mat');
X_test=test.d79;
% the number of observated data in training dataset
n = length(train.d79);
y1=7*ones(1000,1);
y2=9*ones(1000,1);
Y = [y1;y2];
% rng(1); % For reproducibility

%% fit a decision tree with 10-fold cross validation
% Dtree = fitctree(X_train,Y, 'CrossVal','on');
% numBranches = @(x)sum(x.IsBranch);
% mdlDefaultNumSplits = cellfun(numBranches, Dtree.Trained);
% figure;
% histogram(mdlDefaultNumSplits)

MaxNumSplitsList=1:1:37;
ErrorList=ones(37,1)*-1;

% Decision tree
for i=1:37
    max_num_splits=MaxNumSplitsList(i);
    Dtree_cv = fitctree(X_train,Y, 'CrossVal','on','MaxNumSplits',max_num_splits);
    error_DT = kfoldLoss(Dtree_cv);
    ErrorList(i,:)=error_DT;
end
m=min(ErrorList);
idx=find(ErrorList==m);

DtreeFinal=fitctree(X_train,Y,'MaxNumSplits',MaxNumSplitsList(idx));
Y_predicted=predict(DtreeFinal,X_test);
diff=abs(Y_predicted-Y)/2;
DT_err = (sum(diff))/2000
%% Bagged trees
BaggedTree = fitcensemble(X_train,Y,'Method','Bag','CrossVal','on','NumLearningCycles',200);
kflc_bagged=kfoldLoss(BaggedTree,'mode','cumulative');
% error_BaggedT = kflc_bagged(end);
% ErrorList(i,:)=error_BaggedT;

m=min(kflc_bagged);
idx=find(kflc_bagged==m)

BaggedtreeFinal=fitcensemble(X_train,Y,'Method','Bag','NumLearningCycles',idx(1,1));
Y_predicted=predict(BaggedtreeFinal,X_test);
diff=abs(Y_predicted-Y)/2;
DT_err = (sum(diff))/2000

%% Boosted trees
BoostedTree = fitcensemble(X_train,Y,'Method','AdaBoostM1','CrossVal','on','NumLearningCycles',200);
kflc_boo=kfoldLoss(BoostedTree,'mode','cumulative');
% error_BaggedT = kflc_bagged(end);
% ErrorList(i,:)=error_BaggedT;

m=min(kflc_boo);
idx=find(kflc_boo==m)

BoostedtreeFinal=fitcensemble(X_train,Y,'Method','AdaBoostM1','NumLearningCycles',idx(1,1));
Y_predicted=predict(BoostedtreeFinal,X_test);


diff=abs(Y_predicted-Y)/2;
DT_err = (sum(diff))/2000

% BoostedTree = fitcensemble(train.d79,Y,'Method','AdaBoostM1','NumLearningCycles',200,'Kfold',10);
% kflc_boo=kfoldLoss(BoostedTree,'mode','cumulative');
% Error_BooT = kflc_boo(end)
% figure(2)
% plot(kflc_boo,'r.');
% xlabel('Num of Learning Cycles');
% ylabel('Misclassification Rate');