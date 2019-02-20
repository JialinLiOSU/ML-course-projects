%% Define functions for 1-NN and 3-NN
function ind=KNN_3(vec_coord, vec_value)
[r_coord,c_coord]=size(vec_coord);
[r_value,c_value]=size(vec_value);
if c_value~=c_coord
    disp('the dimension of vec_coord and vec_value should be same')
end
ind=zeros(r_value,3);
for k=1:r_value % for each of the 1000 test value
    dist_temp1=999; % smallest dist
    dist_temp2=999;
    dist_temp3=999;
    ind_temp1=1; % ind for least dist
    ind_temp2=1;
    ind_temp3=1;
    for i=1: r_coord % for each of the coord in 2000 points
        dist=0;
        for j=1:c_coord % for each of the axis of one record
            dist=dist+(vec_coord(i,j)-vec_value(k,j)).^2;
        end
        if dist==0
            continue;
        end
        dist=dist.^(0.5);
        if dist< dist_temp1
            dist_temp3=dist_temp2;
            dist_temp2=dist_temp1;
            dist_temp1=dist;
            ind_temp3=ind_temp2;
            ind_temp2=ind_temp1;
            ind_temp1=i;
        elseif dist>=dist_temp1 && dist<dist_temp2
            dist_temp3=dist_temp2;
            dist_temp2=dist;
            ind_temp3=ind_temp2;
            ind_temp2=i;
        elseif dist>=dist_temp2 && dist<dist_temp3
            dist_temp3=dist;
            ind_temp3=i;
        end
    end
ind(k,1)=ind_temp1;
ind(k,2)=ind_temp2;
ind(k,3)=ind_temp3;
end