% Machine Learning HW1 Problem 4
% implement 1-NN and 3-NN classifier from 2000 points 
% generated from two equally weighted spherical gaussians

%% generate points from two equally weighted spherical gaussians
% 2000 points
num_c=1000;
mu1 = [0];
mu2=[3];
sigma = [1];
rng default  % For reproducibility
c1_coord = mvnrnd(mu1,sigma,num_c);
c2_coord = mvnrnd(mu2,sigma,num_c);
c1_label = ones(1000,1);
c2_label = ones(1000,1)*-1;
c1=[c1_coord(:,:) c1_label];
c2=[c2_coord(:,:) c2_label];
C=[c1;c2];
% randomly select 1000 points for testing
inx_r=randi([1,length(C)],1000,1);
% conduct 1NN and 3NN
k1=1;
k2=3;

value=C(inx_r,1); % get the 1000 points location value
ind1=knnsearch(C(:,1),value,'K',k1+1); % 1NN result
ind2=knnsearch(C(:,1),value,'K',k2+1); % 3NN result

label_1=C(ind1(:,2),2);
dif1=(label_1~=C(inx_r,2));
sum_1=sum(dif1); % # predicted error for 1NN

s1=C(ind2(:,2),2);
s2=C(ind2(:,3),2);
s3=C(ind2(:,4),2);
label_2=sign(s1+s2+s3);
dif2=(label_2~=C(inx_r,2));
sum_2=sum(dif2); % # predicted error for 3NN

% figure
% plot(R(:,1),R(:,2),'+')
% mu = [2 3];
% sigma = [1 1.5; 1.5 3];
% rng default  % For reproducibility
% R = mvnrnd(mu,sigma,100);