# -Pose_Based_Multi_View_Invariant_Feature

Currently, The code are uploded to construct the poses for different view of gait images.

First, we need to make the pose directories to train the CNN model so that the model can detect different pose accuratly as much as possible. Run the python file named with: Pose_Directory_Construction.py

The pose directories may consists with frames that are not belong to the pose for which pose-directory has been made, this can be happen because of noise and different sape of persons (remember here we are piking the frames from multiple subjects to make a pose directory). To remove unbelonging frames we need to run two matlab script named as: Average_images.m and Select_Images.m


