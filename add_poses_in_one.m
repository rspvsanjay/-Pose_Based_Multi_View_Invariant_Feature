angles = [0,18,36,54,72,90,108,126,144,162,180];
str1 = '';
for numangle=1:11
    if numangle==1
        str1 = '000';
    else
        if angles(numangle)<100
            str1 = char(strcat('0',int2str(angles(numangle))));
        else
            str1 = int2str(angles(numangle));
        end
    end
    path1 = char(strcat('/DATA/sanjay/VT-GAN-master/CASIA_B/CASIA_B',str1,'degree_Centered_Alinged_PEI_5/'));
    list1 = dir(path1);
    fName1 = {list1.name};
    [~,y1]=size(fName1);
    path1
    y1
    save_path = char(strcat('/DATA/sanjay/VT-GAN-master/CASIA_B/CASIA_B',str1,'degree_Centered_Alinged_PEI_5_to_3/'));
    
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
            for fff_no=3:y3-2
                path4 = char(strcat(path3,fName3(fff_no)));
                path5 = char(strcat(path3,fName3(fff_no+1)));
                path6 = char(strcat(path3,fName3(fff_no+2)));
                image1 = double(imread(path4));
                image2 = double(imread(path5));
                image3 = double(imread(path6));
                energy_image = image1 + image2 + image3;
                max1 = max(energy_image(:));
                energy_image = energy_image/max1;
                if ~exist(char(strcat(save_path,fName1(f_no),'/',fName2(ff_no),'/')),'dir')
                    mkdir(char(strcat(save_path,fName1(f_no),'/',fName2(ff_no),'/')));
                end
                imwrite(energy_image,char(strcat(save_path,fName1(f_no),'/',fName2(ff_no),'/',fName3(fff_no))));
            end
        end
    end
end