% HW3 of Machine Learning Class Problem 3 about K means
clear
rng(123)
train = load('train79.mat');
test = load('test79.mat');
% the number of observated data in training dataset
n = length(train.d79);
y1=ones(1000,1);
y2=(-1)*ones(1000,1);
Y = [y1;y2];
Y_predicted=zeros(2000,1);
k_list=[2,5,10,50];
rate_misc_list=(-1)*ones(4,1);
for i=1:length(rate_misc_list)
    k=k_list(i);
    [idx_class_p,Cent_coord,sumd] = kmeans(train.d79,k); % Conduct K-means
    idx_class_clu=zeros(k,1);
    for j=1:k % for each of the clusters
        idx_ismem=ismember(idx_class_p,j);
        % number of points belong to 7 and 9
        num_7=sum(idx_ismem(1:1000,1));
        num_9=sum(idx_ismem(1001:2000,1));
        if num_7>=num_9
            idx_class=7;
            idx_class_p(ismember(idx_class_p,j))=1;
        else
            idx_class=9;
            idx_class_p(ismember(idx_class_p,j))=-1;
        end
    end
    Num_misc = 1/2*(sum(abs(Y-idx_class_p))); % number of misclassification
    rate_misc=Num_misc/n; % rate of misclassification
    rate_misc_list(i)=rate_misc;
end
rate_misc_list