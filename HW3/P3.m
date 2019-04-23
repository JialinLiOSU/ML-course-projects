% HW3 of Machine Learning Class Problem 3 about K means
train = load('train79.mat');
test = load('test79.mat');
% the number of observated data in training dataset
n = length(train.d79);
y1=ones(1000,1);
y2=(-1)*ones(1000,1);
Y = [y1;y2];

Cent_7=mean(train.d79(1:1000,:));
Cent_9=mean(train.d79(1001:2000,:));
Cent=[Cent_7;Cent_9];
% Centroid point calculated from 7/9 dataset
k_list=[2,5,10,50]
rate_misc_list=(-1)*ones(4,1);
for i=1:length(rate_misc_list)
    k=k_list(i);
    [idx,C,sumd] = kmeans(train.d79,k);
    [Idx,D] = knnsearch(Cent,C,'k',1);
    l2 = find(Idx==2);
    idx_class = idx;
    idx_class(ismember(idx_class,l2))=-1;
    idx_class(idx_class~=-1)=1;
    Num_misc = 1/2*(sum(abs(Y-idx_class)));
    rate_misc=Num_misc/n;
    rate_misc_list(i)=rate_misc;
end
rate_misc_list