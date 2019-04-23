% HW3 of Machine Learning Class Problem 3 about K means
train = load('train79.mat');
test = load('test79.mat');
% the number of observated data in training dataset
n = length(train.d79);

%% PROBLEM.4 K-MEANS %%
% Centroid point calculated from 7/9 dataset
% k = 50;
k_list=[2,5,10,50]
for k=k_list
    [idx,C,sumd] = kmeans(train.d79,k);
    [Idx,D] = knnsearch(mean_data,C,'k',1);
    l2 = find(Idx==2);
    idx_class = idx_;
    idx_class(ismember(idx_class,l2))=-1;
    idx_class(idx_class~=-1)=1;
    MisC = 1/2*(sum(abs(L_data-idx_class)));
end