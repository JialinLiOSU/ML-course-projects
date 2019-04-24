clear
train = load('train79.mat');
train=train.d79;
test = load('test79.mat');
test=test.d79;
label = vertcat(ones(1000,1)*1, ones(1000,1)*-1);
N=2000;

BAT = fitcensemble(train,label,'Method','Bag','NumLearningCycles',200);
testBTResult = (BAT.predict(test));
BTDiff = testBTResult - label;
BTLoss = transpose(BTDiff)*BTDiff/4/N
BTAnotherLoss = sum(abs(BTDiff)/2)/2000
