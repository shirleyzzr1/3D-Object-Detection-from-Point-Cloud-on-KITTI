import numpy as np
import open3d as o3d
import struct
import matplotlib.pyplot as plt
from DBSCAN import DBSCAN
def convert_kitti_bin_to_pcd(binFilePath):
    size_float = 4
    list_pcd = []
    with open(binFilePath, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.array(list_pcd)
    return np_pcd
def ground_segmentation(data):
    #max iterate time
    N = 100 
    #min points to have within the range of the plane
    inlier_limit = 20000  
    #the distance threshold
    Td = 0.5
    potential_plane = []     
    data = np.append(data, np.ones([1, len(data)]).T, axis=1) #extend the point into 4D point
    iter = 0
    while iter<N:
        #randomly choose 3 points
        idx = np.random.choice(len(data),3)
        flat = data[idx,:]
        #using three points to describe a plane
        [[x1,y1,z1,_],[x2,y2,z2,_],[x3,y3,z3,_]] = flat
        A = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
        B = (x3 - x1) * (z2 - z1) - (x2 - x1) * (z3 - z1)
        C = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        D = -(A * x1 + B * y1 + C * z1)
        distance = abs(np.array([A,B,C,D]).dot(data.T))
        count = sum(distance<Td)
        if count>inlier_limit:
            potential_plane.append([A,B,C,D,count])
        iter +=1
    #find the best plane from the potential_plane
    potential_plane = np.array(potential_plane)
    bestplane_idx = np.argmax(potential_plane[:,-1])
    bestplane = potential_plane[bestplane_idx]
    distance = abs(bestplane[:4].dot(data.T))
    ground_idx = np.argwhere(distance<Td)
    ground_idx  = ground_idx.flatten()
    #using least square method to do the calculation again
    A = np.array([[0,0,0],[0,0,0],[0,0,0]])
    b = np.array([0,0,0])
    for i in ground_idx:
        [x,y,z] = data[i,:3]
        A = A+np.array([[x**2,x*y,x],[x*y,y**2,y],[x,y,1]])
        b = b+ np.array([x*z,y*z,z])
    [a0,a1,a2] = np.linalg.inv((A.T).dot(A)).dot(A.T).dot(b)
    distance = abs(np.array([a0,a1,-1,a2]).dot(data.T))
    segmented_idx = np.argwhere(distance>0.8)
    segmented_idx = segmented_idx.flatten()
    segmented_cloud = data[segmented_idx,:3]

    colors = np.array([[1,0,0] for _ in range(len(data))])
    colors[segmented_idx,:] = [0,0,1]

    print('origin data points num:', data.shape[0])
    print('segmented data points num:', segmented_cloud.shape[0])
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data[:,:3])
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    #visualize the ground
    o3d.visualization.draw_geometries([point_cloud])
    np.savetxt("segmented_cloud.txt",segmented_cloud)
    return segmented_cloud

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
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd.cluster_dbscan(eps=1, min_points=100, print_progress=True))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd],
                                    zoom=0.455,
                                    front=[-0.4999, -0.1659, -0.8499],
                                    lookat=[2.1813, 2.0619, 2.0999],
                                    up=[0.1204, -0.9852, 0.1215])
    return labels
def clustering(data):
    # dbscan = DBSCAN()
    # dbscan.fit(data)
    # clusters_index = dbscan.fit_predict(data)
    dbscan = DBSCAN()
    dbscan.fit(data)
    clusters_index = dbscan.predict(data)
    clusters_index = np.array(clusters_index).astype(int)

    return clusters_index

def plot_cluster(data, cluster_index):
    point_cloud = o3d.geometry.PointCloud()
    colors_base = np.array([[55, 126, 184], [255, 127, 0], [77, 175, 74],
                            [247, 129, 191], [166, 86, 40], [152, 78, 163],
                            [153, 153, 153], [228, 26, 28], [222, 222, 0]])
    colors_Len = max(cluster_index) // 9
    colors = np.repeat(colors_base.reshape([1, 27]), colors_Len, axis=0).reshape([9 * colors_Len, 3])
    colors = np.append(colors, colors_base[0:max(cluster_index) % 9, :], axis=0)
    colors = np.append(colors, [[0, 0, 0]], axis=0)  # color of noise
    point_cloud.points = o3d.utility.Vector3dVector(data)
    point_cloud.colors = o3d.utility.Vector3dVector(colors[cluster_index] / 255)
    o3d.visualization.draw_geometries([point_cloud])
if __name__=="__main__":
    point_data = convert_kitti_bin_to_pcd("/home/shirleyzzr/dataset/kitti/object/training/velodyne/007301.bin")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_data[:,:3])
    point_no_ground = faster_ground_removal(pcd)
    index = faster_dbscan(point_no_ground)
    np.savetxt("label.txt",index)
    # cluster_index = clustering(point_data)
    # plot_cluster(point_data,cluster_index)


    # pointcloud = o3d.geometry.PointCloud()
    # pointcloud.points = o3d.utility.Vector3dVector(point_data)
    # o3d.visualization.draw_geometries([pointcloud])
