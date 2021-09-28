path1 = '/DATA/sanjay/VT-GAN-master/CASIA_B/CASIA_B000degree_Centered_Alinged_PEI_5/';
list1 = dir(path1);
fName1 = {list1.name};
[~,y1]=size(fName1);
path1
y1
save_path = '/DATA/sanjay/VT-GAN-master/CASIA_B/CASIA_B000degree_Centered_Alinged_PEI_5_to_5_with_1_skip/';

for f_no=3:y1
    path2 = char(strcat(path1,fName1(f_no),'/'));
    list2 = dir(path2);
    fName2 = {list2.name};
    [~,y2] = size(fName2);
    fName1(f_no)
    
    for ff_no=3:y2
        path3 = char(strcat(path2,fName2(ff_no),'/'));
        list3 = dir(path3);
        fName3 = {list3.name};
        [~,y3] = size(fName3);
        energy_image = double(zeros(size(imread(char(strcat(path3,fName3(3)))))));
        count = 0;
        for fff_no=3:y3
            path4 = char(strcat(path3,fName3(3)));
            path5 = char(strcat(path3,fName3(4)));
            path6 = char(strcat(path3,fName3(5)));
            path7 = char(strcat(path3,fName3(6)));
            path8 = char(strcat(path3,fName3(7)));
            
            image1 = double(imread(path4));
            image2 = double(imread(path5));
            image3 = double(imread(path6));
            image4 = double(imread(path7));
            image5 = double(imread(path8));
            
            if fff_no == 3
                image1 = double(zeros(size(imread(char(strcat(path3,fName3(3)))))));
            end
            if fff_no == 4
                image2 = double(zeros(size(imread(char(strcat(path3,fName3(3)))))));
            end
            if fff_no == 5
                image3 = double(zeros(size(imread(char(strcat(path3,fName3(3)))))));
            end
            if fff_no == 6
                image4 = double(zeros(size(imread(char(strcat(path3,fName3(3)))))));
            end
            if fff_no == 7
                image5 = double(zeros(size(imread(char(strcat(path3,fName3(3)))))));
            end
            
            energy_image = image1 + image2 + image3 + image4 + image5;
            max1 = max(energy_image(:));
            energy_image = energy_image/max1;  
            
            if ~exist(char(strcat(save_path,fName1(f_no),'/',fName2(ff_no),'/')),'dir')
                mkdir(char(strcat(save_path,fName1(f_no),'/',fName2(ff_no),'/')));
            end
            imwrite(energy_image,char(strcat(save_path,fName1(f_no),'/',fName2(ff_no),'/',fName3(fff_no))));
        end
    end
end