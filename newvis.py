from cmath import pi
from cv2 import split
from matplotlib.transforms import Bbox
import numpy as np
import open3d as o3d
from kitti_utils import Kittidata
import os
from torch.utils.data import Dataset
import torch
from model import PointNet
import matplotlib.pylab as plt
from bounding_box import BoundingBox
import cv2 as cv2
import copy
from scipy.spatial.transform import Rotation as R
def faster_ground_removal(data):
    pcd = data
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.3,
                                            ransac_n=3,
                                            num_iterations=1000)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)    
    return outlier_cloud
def faster_dbscan(data):
    pcd = data
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Info) as cm:
        labels = np.array(
            pcd.cluster_dbscan(eps=0.5, min_points=20, print_progress=False))
    # max_label = labels.max()
    # print(f"point cloud has {max_label + 1} clusters")
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return labels
class TestDataset(Dataset):
    def __init__(self,img,clusters,sample_points=1000):
        super(TestDataset, self).__init__()
        self.img = img
        self.clusters = clusters
        self.sample_points = sample_points

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self,idx):
        cluster = self.clusters[idx][0]

        feature = self.img[cluster]

        if len(feature)>1000:
            choice = np.random.choice(len(feature),self.sample_points)
            feature = feature[choice,:]
    
        mean = np.mean(feature,axis=0)
        feature -= mean
        feature /= np.max(np.linalg.norm(feature, axis=1))

        alpha = 2*np.pi*np.random.random()
        rotation_matrix = [[np.cos(alpha),0,np.sin(alpha)],[-np.sin(alpha),np.cos(alpha),0],[0,0,1]]
        feature = feature.dot(rotation_matrix)
        if len(feature)<1000:
            feature = np.r_['0,2',feature,np.zeros([1000-len(feature),3])]

        feature = np.transpose(feature)
        feature = torch.from_numpy(feature.astype(np.float32))
        return feature,idx
def generate_label(labels,scores,bbx_3ds,bbx_2ds,result_path,idx):
    file_path = os.path.join(result_path,str(idx)+".txt")
    catagory = ["Cyclist","Pedestrian","Vehicle","Other"]
    with open(file_path,"a") as f:
        lines = []
        for i in range(len(labels)):
            bbx2d = bbx_2ds[i]
            xmin,ymin = np.min(bbx2d,axis=0)
            xmax,ymax = np.max(bbx2d,axis=0)

            bbx3d = bbx_3ds[i]
            if catagory[int(labels[i])]=="Other":
                continue
            line_list = [
                catagory[int(labels[i])],
                str(-1),
                str(-1),
                str(-3),
                '%.2f' % xmin,'%.2f' % ymin,'%.2f' % xmax,'%.2f' % ymax,
                '%.2f' % bbx3d[0],'%.2f' % bbx3d[1],'%.2f' % bbx3d[2],
                '%.2f' % bbx3d[3],'%.2f' % bbx3d[4],'%.2f' % bbx3d[5],
                '%.2f' % bbx3d[6],
                '%.2f' % scores[i]
            ]
            linestr = " ".join(line_list) + "\n"
            lines.append(linestr)
            linestr = " ".join(line_list) + "\n"
            lines.append(linestr)
        if(len(lines))==0:
            line_list = [
                "don't care",
                str(-1),
                str(-1),
                str(-3),
                '%.2f' % 0,'%.2f' % 0,'%.2f' % 0,'%.2f' % 0,
                '%.2f' % 0,'%.2f' % 0,'%.2f' % 0,
                '%.2f' % 0,'%.2f' %0,'%.2f' % 0,
                '%.2f' % 0,
                '%.2f' % 0
            ]
            linestr = " ".join(line_list) + "\n"
            lines.append(linestr)
        for line in lines:
            f.write(line)
def get_bbx_information(points_in_rect):
    #https://stackoverflow.com/questions/32892932/create-the-oriented-bounding-box-obb-with-python-and-numpy
    a = points_in_rect[:,[0,2]]
    ca = np.cov(points_in_rect[:,[0,2]],y = None,rowvar = 0,bias = 1)
    v, vect = np.linalg.eig(ca)
    rotate = np.linalg.inv(vect)
    ry = np.arctan2(vect[0][1],vect[0][0])
    ar = np.matmul(rotate,a.T).T
    # get the minimum and maximum x and y 
    mina = np.min(ar,axis=0)
    maxa = np.max(ar,axis=0)
    diffxz = (maxa - mina)
    w,l = diffxz[1],diffxz[0]
    centerxz = mina + diffxz/2
    centerxz = np.dot(vect,centerxz)
    # centerxz = np.dot(vect,centerxz)

    h = max(points_in_rect[:,1])-min(points_in_rect[:,1])
    centery = max(points_in_rect[:,1])

    # x_corners = [l/2,-l/2,-l/2,l/2,l/2]
    # z_corners = [w/2,w/2,-w/2,-w/2,w/2]
    # box_corner = np.c_[x_corners,z_corners]

    # rotatey = np.array([[np.cos(ry),np.sin(ry)],[-np.sin(ry),np.cos(ry)]])
    # rotate_box = np.matmul(rotatey,box_corner.T).T

    # rotate_box[:,0] = rotate_box[:,0] + centerxz[0]
    # rotate_box[:,1] = rotate_box[:,1] + centerxz[1]

    # ax = fig.add_subplot(122)
    # ax.scatter(a[:,0],a[:,1])
    # ax.scatter([centerxz[0]],[centerxz[1]])    
    # ax.plot(rotate_box[:,0],rotate_box[:,1],'-')
    # plt.axis('equal')

    # plt.show()
    return h,w,l,centerxz[0],centery,centerxz[1],ry



if __name__=="__main__":
    kitti_dir = "../../../dataset/kitti/object"
    pcd = o3d.geometry.PointCloud()
    split_dir = os.path.join(kitti_dir,"ImageSets","val"+".txt")
    # split_name = np.loadtxt(split_dir)
    split_name = [7301]
    vis_flag = 0
    net = PointNet()
    net.load_state_dict(torch.load("./model/model_72.pth",map_location=torch.device('cpu')))
    for split_idx in split_name:
        print("split_idx",split_idx)
        kitti = Kittidata(kitti_dir,"training",int(split_idx))
        kitti.lidar = kitti.get_lidar_in_image_fov()
        # kitti.show_lidar_with_boxes()
        pcd.points = o3d.utility.Vector3dVector(kitti.lidar)
        pcd = faster_ground_removal(pcd)
        labels = faster_dbscan(pcd)
        #generate the clusters and send them to the network
        clusters = []
        for label in range(max(labels)):
            idx = np.array(np.where(labels==label)).astype(int)
            clusters.append(list(idx))
        clusters = np.array(clusters)
        testdata = TestDataset(np.asarray(pcd.points),clusters)
        testloader= torch.utils.data.DataLoader(testdata,batch_size=1)
        predicted_scores = []
        indexes = []
        labels = []
        for i, data in enumerate(testloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, index = data
            net.eval()
            # inputs, labels = inputs.to(device),labels.to(device)
            outputs = net(inputs)
            score, predicted = torch.max(outputs.data, 1) 
            predicted_scores = np.append(predicted_scores,score)
            labels = np.append(labels,predicted)
            indexes = np.append(indexes,index)
        right=["Cyclist","Pedestrian","Vehicle","Other"]
        colors = [[1,0,0],[0,1,0],[0,0,1],[1,1,0]]
        points = np.asarray(pcd.points)
        color_array = np.zeros([len(np.asarray(pcd.points)),3])
        bbx_3ds = []
        bbx_2ds = []
        lines = [[0,1],[1,2],[2,3],[3,0],[0,4],[1,5],[2,6],[3,7],[4,5],[5,6],[6,7],[7,4]]

        image = kitti.image
        line_sets = []
        for i in range(len(indexes)):
            cluster = clusters[int(indexes[i])][0]
            label = labels[i]
            color = colors[int(label)]
            color_array[cluster] = color
            #first project the lidar from velo to camera rect plane
            points_in_rect = kitti.project_velo_to_rect(points[cluster])
            bbx_segments = get_bbx_information(np.array(points_in_rect))
            bbx_3d = np.zeros(15)
            bbx_3d[8:15] = bbx_segments
            bbx_3d_velo = kitti.get_bbx(bbx_3d)
            bbx_3ds.append(bbx_segments)
            #get the bbx in image
            bbx_2d = kitti.project_velo_to_image(bbx_3d_velo)
            bbx_2ds.append(bbx_2d)
            if vis_flag:
                for l in lines:
                    cv2.line(image, (int(bbx_2d[l[0]][0]), int(bbx_2d[l[0]][1])), \
                    (int(bbx_2d[l[1]][0]), int(bbx_2d[l[1]][1])), [255*color[2],255*color[1],255*color[0]], 2)
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(bbx_3d_velo)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)
                line_sets.append(line_set)
        pcd.colors = o3d.utility.Vector3dVector(color_array)
        generate_label(labels,predicted_scores,bbx_3ds,bbx_2ds,"../dataset/generate_label/","%06d"%int(split_idx))
        if vis_flag:
            o3d.visualization.draw_geometries([pcd,*line_sets])
            cv2.imshow("image",image)
            cv2.waitKey(0)


