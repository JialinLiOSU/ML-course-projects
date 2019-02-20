%% Define functions for 1-NN and 3-NN
function ind=KNN_1(vec_coord, vec_value)
[r_coord,c_coord]=size(vec_coord);
[r_value,c_value]=size(vec_value);
if c_value~=c_coord
    disp('the dimension of vec_coord and vec_value should be same')
end
ind=zeros(r_value,1);
for k=1:r_value % for each of the 1000 test value
    dist_temp=999;
    ind_temp=1;
    for i=1: r_coord % for each of the coord in 2000 points
        dist=0;
        for j=1:c_coord % for each of the axis of one record
            dist=dist+(vec_coord(i,j)-vec_value(k,j)).^2;
        end
        if dist==0
            continue;
        end
        dist=dist.^(0.5);
        if dist< dist_temp
            dist_temp=dist;
            ind_temp=i;
        end
    end
ind(k)=ind_temp;
end