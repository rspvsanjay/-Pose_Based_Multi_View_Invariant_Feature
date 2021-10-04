from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import pca
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import cv2
import os
import numpy as np

def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

rank = 10
def classifyMe(path1,gallary,probe):    
    subjects = os.listdir(path1)
    subjectsNumber = len(subjects)
    
    X1 = []
    y1 = []
    pid = 0# if number of subjects present from 001 to 124 then pid = 63 otherwise pid=0 if only testing data are present
    for number1 in range(pid,subjectsNumber):
        if (number1+63)<100:
            sub1 = '0' + str(number1+63)
        else:
            sub1 = str(number1+63)
        path5 = path1 + sub1 + '/'

        for sequence in ['nm-01', 'nm-02', 'nm-03', 'nm-04']:
            path9 = path5 + sequence + '/' + gallary + '/'
            poses = os.listdir(path9)
            posesNumber = len(poses) 
            for number2 in range(0,posesNumber):
                path13 = path9 + poses[number2]
                img = cv2.imread(path13, 0)
                img = img.flatten().astype(np.float32)
                X1.append(img) 
                y1.append(number1+1)

    X1 = np.asarray(X1)
    y1 = np.asarray(y1).astype(np.int32)

    pca_model1 = pca.PCA(n_components=int(min(X1.shape)*0.2), whiten=False)
    pca_model1.fit(X1)
    X1 = pca_model1.transform(X1)
    lda_model1 = LinearDiscriminantAnalysis(n_components=45)
    lda_model1.fit(X1, y1)
    X1 = lda_model1.transform(X1)
    nbrs1 = KNeighborsClassifier(n_neighbors=rank*2, weights='distance', metric='euclidean')
    nbrs1.fit(X1, y1)

    subjects = os.listdir(path1)
    subjectsNumber = len(subjects)
    
    testy = []
    testX1 = []
    for number1 in range(pid,subjectsNumber):#            
        if (number1+63)<100:
            sub1 = '0' + str(number1+63)
        else:
            sub1 = str(number1+63)
        path17 = path1 + sub1 + '/'

        for sequence in ['nm-05', 'nm-06']:
            testy.append(number1)
            path21 = path17 + sequence + '/' + probe + '/'
            poses = os.listdir(path21)  
            path21 = path21 + poses[0]  
            img = cv2.imread(path21, 0)
            img = img.flatten().astype(np.float32) 
            testX1.append(img)

    testX1 = np.asarray(testX1).astype(np.float32)
    tX = pca_model1.transform(testX1)
    tX = lda_model1.transform(tX)
    pred1 = nbrs1.predict_proba(tX)
    return pred1, testy

rank_result = np.zeros((rank, 11, 11))
index11 = -1
index22 = -1
angles_gallary = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
for gallary in angles_gallary:
    path1 = '/DATA/sanjay/VT-GAN-master/generated_from_feature1/Generated_Energy_Image_'+gallary+'/'
    path2 = '/DATA/sanjay/VT-GAN-master/generated_from_feature2/Generated_Energy_Image_'+gallary+'/'
    path3 = '/DATA/sanjay/VT-GAN-master/generated_from_feature3/Generated_Energy_Image_'+gallary+'/'
    path4 = '/DATA/sanjay/VT-GAN-master/generated_from_feature4/Generated_Energy_Image_'+gallary+'/'
    path5 = '/DATA/sanjay/VT-GAN-master/generated_from_feature5/Generated_Energy_Image_'+gallary+'/'

    path = []
    path.append(path1)
    path.append(path2)
    path.append(path3)
    path.append(path4)
    path.append(path5)
    angles_probe = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
    index11 = index11 + 1
    for probe in angles_probe:
        index22 = index22 + 1
        testy = []
        predicted_prob = []
        for num1 in range(0,len(path)):        
            pred, testy = classifyMe(path[num1],gallary,probe)        
            predicted_prob.append(pred)
        predicted_class_rnk = []
        for numm1 in range(0,rank):
            num1, num2 = predicted_prob[0].shape
            predicted_class = []
            for number4 in range(0,num1):
                class1 = []
                prob1 = []
                sum_prob = []

                for num3 in range(0,len(path)):
                    temp1 =  sorted(predicted_prob[num3][number4],reverse=True) 
                    index1 = np.where(predicted_prob[num3][number4] == temp1[numm1]) 
                    index1 = np.asarray(index1)  
                    class1.append(index1[0][0])  
                    prob1.append(temp1[numm1])

                unique_class1 = unique(class1)
                for num11 in range(0,len(unique_class1)):
                    sum_prob.append(0)

                for num11 in range(0,len(unique_class1)):
                    for num22 in range(0,len(class1)):
                        if unique_class1[num11]==class1[num22]:
                            sum_prob[num11] = sum_prob[num11] + prob1[num22]

                maxval = np.amax(sum_prob)
                index = -1
                for num2 in range(0,len(sum_prob)):
                    if sum_prob[num2] == maxval:
                        index = num2 
                predicted_class.append(unique_class1[index])
            predicted_class_rnk.append(predicted_class)

        print("Gallary angle: ",gallary)
        print("Probe angle: ", probe)
        for number21 in range(0,rank):
            count1 = 0
            for num1 in range(0,len(testy)):
                k=0
                for num2 in range(0,number21+1):
                    if k==0:
                        if testy[num1] == predicted_class_rnk[num2][num1]:
                            count1 = count1+1
                            k=1
                            break
            print("Accuracy of Rank",str(number21+1),": ",(float(count1)/float(len(testy)))*100) 
            rank_result[number21][index11][index22] = round((float(count1)/float(len(testy)))*100, 2)

        if index22 == 10:
            index22 = -1
        print('--------------------------------------------------------------')  
        print('--------------------------------------------------------------')  
        print('--------------------------------------------------------------') 

for num1 in range(0,rank):
    print(np.mean(rank_result[num1], axis=0))
    pathh = '/DATA/sanjay/VT-GAN-master/divers_feature_and_view_nm2nm_with_rank_of_' + str(num1+1) + '.csv'
    np.savetxt(pathh,rank_result[num1])


