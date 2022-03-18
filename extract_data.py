import numpy as np
from kitti_utils import Kittidata
import os
import open3d as o3d
import matplotlib.pylab as plt
# from sklearn.cluster import DBSCAN
def faster_ground_removal(data):
    """
    remove the ground
    """
    pcd = data
    # o3d.visualization.draw_geometries([pcd])
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.2,
                                            ransac_n=3,
                                            num_iterations=1000)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)    
    return outlier_cloud
def faster_dbscan(data):
    """
    Clustring using DBSCAN
    """
    pcd = data
    # o3d.visualization.draw_geometries([pcd])

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Info) as cm:
        labels = np.array(
            pcd.cluster_dbscan(eps=1, min_points=50, print_progress=False))
    # max_label = labels.max()
    # print(f"point cloud has {max_label + 1} clusters")
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd],
    #                                 zoom=0.455,
    #                                 front=[-0.4999, -0.1659, -0.8499],
    #                                 lookat=[2.1813, 2.0619, 2.0999],
    #                                 up=[0.1204, -0.9852, 0.1215])
    return labels
def extract_data(kitti_cls_dir,kitti_dir,split):
    """
    extract data by catagory, "Vehicle","Pedestrain","Cyclist","Other"
    "Other" object are result of clustering
    """
    os.makedirs(os.path.join(kitti_cls_dir,split,"Vehicle"),exist_ok=True)
    os.makedirs(os.path.join(kitti_cls_dir,split,"Pedestrian"),exist_ok=True)
    os.makedirs(os.path.join(kitti_cls_dir,split,"Cyclist"),exist_ok=True)
    os.makedirs(os.path.join(kitti_cls_dir,split,"Other"),exist_ok=True)
    split_dir = os.path.join(kitti_dir,"ImageSets",split+".txt")
    split_name = np.loadtxt(split_dir)
    Vehicle_cnt,Pedestrain_cnt,Cyclist_cnt,Other_cnt = 0,0,0,0
    #first generate the data from the groundtruth
    for i in range(len(split_name)):
        print(i)
        kitti = Kittidata(kitti_dir,"training",split_name[i])
        kitti.lidar = kitti.get_lidar_in_image_fov()
        # kitti.show_lidar_with_boxes()
        inliers = np.zeros(len(kitti.lidar),dtype=bool)
        pcd = o3d.geometry.PointCloud()
        for label in kitti.label_gt:
            inlier = kitti.in_hull(kitti.lidar,label["bbx"])
            # print("label",label['label'])
            if label['label']=="Vehicle":
                np.savetxt(os.path.join(kitti_cls_dir,split,label["label"],str(Vehicle_cnt)+".txt"),kitti.lidar[np.where(inlier==True)])
                Vehicle_cnt +=1
            elif label['label']=='Pedestrian':
                np.savetxt((os.path.join(kitti_cls_dir,split,label["label"],str(Pedestrain_cnt)+".txt")),kitti.lidar[np.where(inlier==True)])
                Pedestrain_cnt +=1
            elif label['label']=="Cyclist":
                np.savetxt(os.path.join(kitti_cls_dir,split,label["label"],str(Cyclist_cnt)+".txt"),kitti.lidar[np.where(inlier==True)])
                Cyclist_cnt+=1
            inlier = kitti.in_hull(kitti.lidar,label["bbx"])
            inliers = inliers | inlier
        pcd.points = o3d.utility.Vector3dVector(kitti.lidar[np.where(inliers==False)])
        # o3d.visualization.draw_geometries([pcd])

        #remove all the pointcloud inside the bounding box
        gt_remove_cloud = pcd
        # gt_remove_cloud = pcd.select_by_index((list(gt_box_idx.astype(int))), invert=True)    
        #remove the ground
        # print("start_removing_ground")
        ground_remove_cloud = faster_ground_removal(gt_remove_cloud)
        #cluster over the rest of the pointcloud
        # print("start_clustring")

        labels = faster_dbscan(ground_remove_cloud)
        lidar_points = np.asfarray(ground_remove_cloud.points)
        for label in range(max(labels)):
            idx = np.where(labels==label)
            idx = list(idx)
            # pointcloud = o3d.geometry.PointCloud()
            # pointcloud.points = o3d.utility.Vector3dVector(lidar_points[tuple(idx)])
            # o3d.visualization.draw_geometries([pointcloud])
            np.savetxt(os.path.join(kitti_cls_dir,split,"Other",str(Other_cnt)+".txt"),lidar_points[tuple(idx)])

            Other_cnt+=1


if __name__=="__main__":
    kitti_cls_dir = "./data"
    kitti_dir = "../../../dataset/kitti/object"
    extract_data(kitti_cls_dir,kitti_dir,"val")
    
