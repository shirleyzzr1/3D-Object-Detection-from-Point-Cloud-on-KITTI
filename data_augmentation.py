"""
Augmented training data by removing pointclouds which is less than 
a certain threshold, also only save 5000 images for Other catagory
to balance the number of training data
"""
from dataset import KittiDataset
import os
import numpy as np
if __name__=="__main__":
    # train_data = KittiDataset("./data/",split="training")
    data_path = "./data"
    split = "training"
    cataname = np.loadtxt(os.path.join(data_path,split,"kitti.txt"),delimiter=" ",dtype=np.str)
    filenames=[]
    other_count = 0
    for i in range(len(cataname)):
        name = cataname[i]
        num = name.split("_")[-1]
        cata = name.split("_"+num)[0]
        filename = os.path.join(data_path,split,cata,num)
        point_data_len = len(np.loadtxt(filename))
        if (cata=="Cyclist" or "Pedestrian" or "Other") and point_data_len<20:continue
        if (cata=="Vehicle") and point_data_len<100:continue
        if (cata=="Other"):
            other_count +=1
            if other_count>5000:break
        # filenames = np.append(filenames,cataname[i]+str(point_data_len))
        filenames = np.append(filenames,cataname[i])
        print(i)

    np.savetxt(os.path.join(data_path,split,"kitti_augmented.txt"),filenames,fmt="%10s",delimiter=" ")




