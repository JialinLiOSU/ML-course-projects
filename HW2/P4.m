% HW2 of Machine Learning Class Problem 3
train = load('train79.mat');
test = load('test79.mat');
% the number of observated data in training dataset
[n,d]= size(train.d79);
train_size=[25,100,200,500,1000];
it_list=[0:200:2000];
l_it=length(it_list);
l=length(train_size);
%% PCA reduce the dimension
train_coeff = pca(train.d79,'NumComponents', 400);
test_coeff= pca(test.d79,'NumComponents', 400);
train_reduced=train.d79*train_coeff;
test_reduced=test.d79*test_coeff;
ls_err_list=ones(l,l_it);
for i=1:l
    train_rn=train_reduced([1:train_size(i) 1001:1000+train_size(i)],:);
    test_rn=test_reduced([1:train_size(i) 1001:1000+train_size(i)],:);
    for j=1:l_it
        %% least squares linear classifier
        Y_ls=[ones(train_size(i),1);-1*ones(train_size(i),1)];
        X_ls=[ones(train_size(i)*2,1),train_rn];
        theta=zeros(400+1,1);

        alpha = 1*10e-11;
        computeCost(X_ls, Y_ls, theta);
        theta = gradientDescent(X_ls, Y_ls, theta, alpha, it_list(j));
        X_ls_test=[ones(train_size(i)*2,1),test_rn];
        LS_label=sign(X_ls_test*theta);
        LS_err=1/2*(sum(abs(Y_ls-LS_label)))/(2*train_size(i));

        ls_err_list(i,j)=LS_err;
    end
end

plot(ls_err_list(1,:),'r')
hold on
plot(ls_err_list(2,:),'g')
hold on
plot(ls_err_list(3,:),'b')
hold on
plot(ls_err_list(4,:),'y')
hold on
plot(ls_err_list(5,:),'k')
legend('25','100','200','500','1000','Location','NE');
title('Error rate of linear regression with different number of iteration')
xticks([1:2:11])
xticklabels({'0','400','800','1200','1600','2000'})
xlabel('number of iteration');
ylabel('Error rate');