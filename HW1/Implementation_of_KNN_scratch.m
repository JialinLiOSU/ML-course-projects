% Machine Learning HW1 Problem 4
% implement 1-NN and 3-NN classifier from scratch but not use knnsearch
% generated from two equally weighted spherical gaussians


num_c=1000;
num_t=1000;

err_rate_1nn=zeros(11,1);
err_rate_3nn=zeros(11,1);


%% Compute the error rate for different dimension p
count=1;
for p=1:10:101
    %% generate points from two equally weighted spherical gaussians
    % 2000 points
    mu1 = zeros(p,1);
    mu2=zeros(p,1);
    mu2(1)=3;
    
    sigma = eye(p,p);
    rng default  % For reproducibility
    c1_coord = mvnrnd(mu1,sigma,num_c);
    c2_coord = mvnrnd(mu2,sigma,num_c);
    c1_label = ones(1000,1);
    c2_label = ones(1000,1)*(-1);
    c1=[c1_coord(:,:) c1_label];
    c2=[c2_coord(:,:) c2_label];
    C=[c1;c2];
    % randomly select 1000 points for testing
    inx_r=randi([1,length(C)],1000,1);
    
    %% conduct 1NN and 3NN
    k1=1;
    k2=3;

    value=C(inx_r,1:p); % get the 1000 points location value
    ind1=KNN_1(C(:,1:p),value); % 1NN result
    ind2=KNN_3(C(:,1:p),value); % 3NN result
    
%     [m,n]=size(ind1);
    label_1=C(ind1(:,1),p+1);
    dif1=(label_1~=C(inx_r,p+1));
    sum_1=sum(dif1); % # predicted error for 1NN

    s1=C(ind2(:,1),p+1);
    s2=C(ind2(:,2),p+1);
    s3=C(ind2(:,3),p+1);
    label_2=sign(s1+s2+s3);
    dif2=(label_2~=C(inx_r,p+1));
    sum_2=sum(dif2); % # predicted error for 3NN
    %% compute error rate and store in arrays
    err_rate_1nn(count)=sum_1/num_t;
    err_rate_3nn(count)=sum_2/num_t;
    count=count+1;
end

%% Figure plot
figure
x=1:11;
plot(x,err_rate_1nn,'-r',x,err_rate_3nn,'--b');
ylim([0 1]);
legend('1-NN','3-NN');


