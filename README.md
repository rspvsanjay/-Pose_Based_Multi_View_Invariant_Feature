# Pose_Based_Multi_View_Invariant_Feature

Currently, The code are uploded to construct the 5 poses for different view of gait images then to get pose feature with high density and low noisy we will get 3 poses for each sequence. The sample data given here that is made one for one angle for other angle we need to edit the code or structure the directories as given in sample data otherwise these code will not work.

First, we need to make the pose directories to train the CNN model so that the model can detect different pose accuratly as much as possible. Run the python file named with: Pose_Directory_Construction.py

The pose directories may consists with frames that are not belong to the pose for which pose-directory has been made, this can be happen because of noise and different sape of persons (remember here we are piking the frames from multiple subjects to make a pose directory). To remove unbelonging frames we need to run two matlab script named as: Average_images.m and Select_Images.m

After refining the pose directories we are ready to train the CNN model and save it. The model will serve as pose detector to create the pose based energy images. Run the python file to train CNN model: Construct_PEI_Using_CNN_Model.py

Then we can use saved CNN model to make the pose based energy image using the following python file: Pose_Directory_Construction.py

To get final pose feature run the MATLAB code named as: add_poses_in_one.m

And we can get anothor feature after running the MATLAB code: PEI_Construction_After_1_skip.m, The feature after running this file can not be said as pose based feature because the feature not satisfiying the pose based energy image definition.


