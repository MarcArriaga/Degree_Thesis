# Position determination using computer vision to determine the focus of attention

**Author:** Marc Arriaga Hidalgo

**Degree:** Bachelor's degree in Audiovisual Systems

**Year:** 2020

This is the code used for my dregree's thesis at Universitat Politècnica de Catalunya(UPC)

## Paths of the database

First of all the code in the file **create_txt.py** runs through the all the trajectories in the database to gather the path along with the yaw, pitch and roll angle of the head and the x, y positons of the person in the frame.

It saves a txt file for each trajectory, all the txt files are stored in the **paths** folder. 

The **split.py** file, stored in the paths folder, creates a txt file for the training and validation process and stores it in the **splits** folder.
