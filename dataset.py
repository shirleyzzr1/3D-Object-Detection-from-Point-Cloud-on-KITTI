import torch
import numpy as np
import open3d as o3d
import os
from torch.utils.data import Dataset
#create Dataset class to perform on all the data here

class KittiDataset(Dataset):
    def __init__(self,img_dir,split="training",sample_points=1000,augment_file = "kitti_augmented.txt"):
        super(KittiDataset, self).__init__()
        self.img_dir = img_dir
        self.features = []
        self.labels = []
        self.classes = ["Cyclist","Pedestrian","Vehicle","Other"]
        self.num = [642,1912,7836,5000]
        self.filenames=[]
        self.split = split
        self.augment_file = augment_file
        self.sample_points = sample_points
        # self.load_data(img_dir)
        self.generate_filename()
    def generate_filename(self):
        self.filenames = np.loadtxt(os.path.join(self.img_dir,self.split,self.augment_file),dtype=np.str)
        # for cata in self.classes:
        #     filelist = os.listdir(os.path.join(self.img_dir,self.split,cata))
        #     filename = np.array([cata+"_"+file for file in filelist])
        #     self.filenames = np.append(self.filenames,filename)
        # print(os.path.join(self.img_dir,self.split,"kitti.txt"))
        # print(self.filenames)
        # np.savetxt(os.path.join(self.img_dir,self.split,"kitti.txt"),self.filenames,fmt="%10s",delimiter=" ")


    def __len__(self):
        return len(self.filenames)

    def read_txt(self,file):
        with open(file, 'r') as infile:
            feature_name = [line.strip() for line in infile]
        return feature_name

    def __getitem__(self,idx):
        # feature = self.feature_name[idx]
        # # print(feature)
        # label = self.labels[idx]
        name = self.filenames[idx]
        num = name.split("_")[-1]
        cata = name.split("_"+num)[0]
        filename = os.path.join(self.img_dir,self.split,cata,num)
        #the first three column is the actual position of the point
        feature = np.loadtxt(filename,delimiter=" ")
        label = self.classes.index(cata)
        # pointcloud = o3d.geometry.PointCloud()
        # pointcloud.points = o3d.utility.Vector3dVector(feature)
        # o3d.visualization.draw_geometries([pointcloud])
        
        #randomly sample some points if sample_num>1000
        #else add zero to all the points
        if len(feature)>1000:
            choice = np.random.choice(len(feature),self.sample_points)
            feature = feature[choice,:]
        # print("filename:",filename,"len",len(feature))
        #calculate means
        mean = np.mean(feature,axis=0)
        feature -= mean
        feature /= np.max(np.linalg.norm(feature, axis=1))
        # pointcloud = o3d.geometry.PointCloud()
        # pointcloud.points = o3d.utility.Vector3dVector(feature)
        # o3d.visualization.draw_geometries([pointcloud])

        #add some rotation on z axis
        alpha = 2*np.pi*np.random.random()
        rotation_matrix = [[np.cos(alpha),0,np.sin(alpha)],[-np.sin(alpha),np.cos(alpha),0],[0,0,1]]
        feature = feature.dot(rotation_matrix)

        # pointcloud = o3d.geometry.PointCloud()
        # pointcloud.points = o3d.utility.Vector3dVector(feature)
        # o3d.visualization.draw_geometries([pointcloud])

        #add some random jitter
        # feature += np.random.normal(0, 0.02, size=feature.shape)  # random jitter
        # #transform the data format
        # pointcloud = o3d.geometry.PointCloud()
        # pointcloud.points = o3d.utility.Vector3dVector(feature)
        # o3d.visualization.draw_geometries([pointcloud])
        if len(feature)<1000:
            feature = np.r_['0,2',feature,np.zeros([1000-len(feature),3])]

        feature = np.transpose(feature)
        feature = torch.from_numpy(feature.astype(np.float32))
        return feature,label


if __name__ =="__main__":

    #visualize one data point
    dataset2 = KittiDataset("./data",split="training")
    print(len(dataset2))
    feature,label = dataset2[773]
    for i in range(10391,15391):
        feature,label = dataset2[i]

    print(label)





    