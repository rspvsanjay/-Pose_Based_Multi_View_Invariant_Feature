path = '/DATA/sanjay/VT-GAN-master/CASIA_B/CASIA_B_pose_data_to_train_CNN/CASIA_B162degree_Centered_Alinged_Pose_Directory_with_length_5_Average/';
list = dir(path);
fName = {list.name};
[~,y]=size(fName);
poses = cell(0,0);
for pose_no=3:y
    poses{pose_no-2}=double(rgb2gray(imread(char(strcat(path,fName(pose_no))))));
    image = poses{pose_no-2};
    max1 = max(image(:));
    image = image/max1;
    poses{pose_no-2} = image;
end

path1 = '/DATA/sanjay/VT-GAN-master/CASIA_B/CASIA_B_pose_data_to_train_CNN/CASIA_B162degree_Centered_Alinged_Pose_Directory_with_length_5/';
save_path = '/DATA/sanjay/VT-GAN-master/CASIA_B/CASIA_B_pose_data_to_train_CNN/CASIA_B162degree_Centered_Alinged_Pose_Directory_with_length_5_selected/';
list1 = dir(path1);
fName1 = {list1.name};
[~,y1]=size(fName1);
save_path

for f_no=3:y1
    path2=char(strcat(path1,fName1(f_no),'/'));
    list2 = dir(path2);
    fName2 = {list2.name};
    [~,y2]=size(fName2);
    fName1(f_no)
    cr = double([]);
    for ff_no=3:y2
        path3 = char(strcat(path2,fName2(ff_no)));
        image = double(rgb2gray(imread(path3)));
        r=corr2(poses{f_no-2},image);
        cr = [cr,r];
    end
    cr2 = double([]);
    for num=1:length(cr)
        if isnan(cr(num))
        else
            cr2 = [cr2,cr(num)];
        end
    end

    mean_cr = sum(cr2)/length(cr2);
    
    if ~exist(char(strcat(save_path,fName1(f_no),'/')),'dir')
        mkdir(char(strcat(save_path,fName1(f_no),'/')));
    end
    for ff_no=3:y2
        path3 = char(strcat(path2,fName2(ff_no)));
        image = double(rgb2gray(imread(path3)));
        max1 = max(image(:));
        image = image/max1;
        if cr(ff_no-2)>=mean_cr
            imwrite(image,char(strcat(save_path,fName1(f_no),'/',fName2(ff_no))));
        end
    end
end