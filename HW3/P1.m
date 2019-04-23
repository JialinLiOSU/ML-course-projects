% HW3 of Machine Learning Class Problem 1
train = load('train79.mat');
test = load('test79.mat');
% the number of observated data in training dataset
n = length(train.d79);
y1=7*ones(1000,1);
y2=9*ones(1000,1);
Y = [y1;y2];

%% fit a decision tree with 10-fold cross validation
Dtree = fitctree(train.d79,Y, 'CrossVal','on');
view(Dtree.Trained{1},'Mode','graph');
rng(123); % For reproducibility
error_DT = kfoldLoss(Dtree)

%% Bagged trees
BaggedTree = fitcensemble(train.d79,Y,'Method','Bag','NumLearningCycles',200,'Kfold',10);
kflc_bagged=kfoldLoss(BaggedTree,'mode','cumulative');
Error_BaggedT = kflc_bagged(end)
figure(1)
plot(kflc_bagged,'r.');
title('performance of bagged decision trees');
xlabel('Num of Learning Cycles');
ylabel('Misclassification Rate');

%% Boosted trees
BoostedTree = fitcensemble(train.d79,Y,'Method','AdaBoostM1','NumLearningCycles',200,'Kfold',10);
kflc_boo=kfoldLoss(BoostedTree,'mode','cumulative');
Error_BooT = kflc_boo(end)
figure(2)
plot(kflc_boo,'r.');
xlabel('Num of Learning Cycles');
ylabel('Misclassification Rate');