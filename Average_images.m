path1 = '/DATA/sanjay/VT-GAN-master/CASIA_B/CASIA_B_pose_data_to_train_CNN/CASIA_B180degree_Centered_Alinged_Pose_Directory_with_length_5/';
save_path='/DATA/sanjay/VT-GAN-master/CASIA_B/CASIA_B_pose_data_to_train_CNN/CASIA_B180degree_Centered_Alinged_Pose_Directory_with_length_5_Average/';
list1 = dir(path1);
fName1 = {list1.name};
[~,y1]=size(fName1);
save_path
mkdir(save_path)
for f_no=3:y1
    path2 = char(strcat(path1,fName1(f_no),'/'));
    list2 = dir(path2);
    fName2 = {list2.name};
    [~,y2] = size(fName2);
    fName1(f_no)
    images = double(zeros(size(imread(char(strcat(path2,fName2(3)))))));
    for ff_no=3:y2
        path3 = char(strcat(path2,fName2(ff_no)));
        image = double(imread(path3));
        images = images + image;
    end
    max1 = max(images(:));
    images = images/max1;
    imwrite(images,char(strcat(save_path,fName1(f_no),'.png')));
end