% HW3 of Machine Learning Class Problem 3 about K means
clear
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
    [idx_class_p,Cent_coord,sumd] = kmeans(train.d79,k); % Conduct K-means
    [idx_class_cluster,D] = knnsearch(Cent,Cent_coord,'k',1); % classify the k clusters to 2 classes 7 or 9
    l2 = find(idx_class_cluster==2); % find the index of cluster who is classified to class 9
    idx_class_p(ismember(idx_class_p,l2))=-1; % assign value of -1 to the points who are class 9
    idx_class_p(idx_class_p~=-1)=1; % assign value of 1 to the other points
    Num_misc = 1/2*(sum(abs(Y-idx_class_p))); % number of misclassification
    rate_misc=Num_misc/n; % rate of misclassification
    rate_misc_list(i)=rate_misc;
end
rate_misc_list