"""
https://github.com/charlesq34/frustum-pointnets/blob
/2ffdd345e1fce4775ecb508d207e0ad465bcca80/kitti/kitti_object.py
"""
import numpy as np
import modern_robotics as mr
import os
import cv2
import struct
import open3d as o3d

class Kittidata:
    def __init__(self,dataset_dir,split,index) -> None:
        self.calib_dir = os.path.join(dataset_dir,split,"calib")
        self.image_dir = os.path.join(dataset_dir,split,"image_2")
        self.lidar_dir = os.path.join(dataset_dir,split,"velodyne")
        self.label_dir = os.path.join(dataset_dir,split,"label_2")

        self.calib = self.read_calib_file(index)
        self.image = self.read_image_file(index)
        self.lidar = self.read_lidar_file(index)
        self.label = self.read_label_file(index)
        self.label_gt = self.label_gt_generate()

        self.image_height,self.image_width = len(self.image),len(self.image[0])


    def read_calib_file(self, index):
        """ Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """
        data = {}
        calib_filename = os.path.join(self.calib_dir, "%06d.txt" % (index))
        # print("calib_filename",calib_filename)
        with open(calib_filename, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        
        self.P = data["P2"]
        self.P = np.reshape(self.P,[3,4])
        self.R0 = data["R0_rect"]
        self.R0 = np.reshape(self.R0,[3,3])
        self.V2C = data["Tr_velo_to_cam"]
        self.V2C = np.reshape(self.V2C,[3,4])
        self.C2V = mr.TransInv(self.V2C)[0:3,:]
        self.f_u = self.P[0,0]
        self.c_u = self.P[0,2]
        self.f_v = self.P[1,1]
        self.c_v = self.P[1,2]
        self.b_x = -self.P[0,3]/self.f_u
        self.b_y = -self.P[1,3]/self.f_u
        return data

    def read_image_file(self,index):
        image_filename = os.path.join(self.image_dir, "%06d.png" % (index))
        img = np.array(cv2.imread(image_filename))
        return img
    
    def read_lidar_file(self,index):
        lidar_filename = os.path.join(self.lidar_dir,"%06d.bin" % (index))
        size_float = 4
        list_pcd = []
        with open(lidar_filename, "rb") as f:
            byte = f.read(size_float * 4)
            while byte:
                x, y, z, intensity = struct.unpack("ffff", byte)
                list_pcd.append([x, y, z])
                byte = f.read(size_float * 4)
        np_pcd = np.array(list_pcd)
        return np_pcd

    def read_label_file(self,index):
        label_data = []
        label_filename = os.path.join(self.label_dir,"%06d.txt" % (index))
        with open(label_filename, "r") as f:
            for line in f.readlines():
                label_data.append(line.split(" "))
        label_data = np.array(label_data)
        return label_data

    def get_lidar_in_image_fov(self):
        """
            project the lidar pointclouds to 2D image
        """
        pts2d = self.project_velo_to_image(self.lidar)
        # np.savetxt("pts2",pts2d)
        fov = (pts2d[:,0]<self.image_width) & (pts2d[:,0] >=0) &\
                (pts2d[:,1]<self.image_height)&(pts2d[:,1]>=0)
        fov = fov & (self.lidar[:,0]>=2)
        lidar_in_fov = self.lidar[fov,:]
        return lidar_in_fov

    def label_gt_generate(self):
        """
            generate groundtruth label for "Vehicle","Pedestrain","Cyclist"
        """
        gt_list = []
        for label in self.label:
            gt_dict={}
            if label[0] in ["Misc","Person_sitting","DontCare"]:continue
            if label[0] in ["Car","Van","Truck","Tram"]:
                gt_dict["label"]="Vehicle"
                gt_dict["bbx"] = self.get_bbx(label)
            else:
                gt_dict["label"]=label[0]
                gt_dict["bbx"] = self.get_bbx(label)
            # inlier = self.in_hull(self.lidar,gt_dict["bbx"])
            # gt_dict["lidar_idx"] = np.where(inlier==True)
            gt_list.append(gt_dict)
        return gt_list

    def show_lidar_with_boxes(self):
        self.lidar = self.get_lidar_in_image_fov()
        print("lidar_len",len(self.lidar))
        pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(self.lidar))
        pcd.paint_uniform_color([0, 0, 0])
        #the points to match
        # lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],\
        #      [0, 4], [1, 5], [2, 6], [3, 7]]
        lines = [[0,1],[1,2],[2,3],[3,0],[0,4],[1,5],[2,6],[3,7],[4,5],[5,6],[6,7],[7,4]]
        colors = [[1, 0, 0] for _ in range(len(lines))]
        line_sets = []
        colors = [[1,0,0],[0,1,0],[0,0,1],[1,1,0]]

        for label in self.label_gt:
            bbx = label["bbx"]
            inlier = self.in_hull(self.lidar,bbx)
            inlier_indices = np.where(inlier==True)
            pcd_colors = np.asarray(pcd.colors)
            if label["label"]=="Cyclist":
                pcd_colors[inlier_indices] = [1,0,0]
            elif label["label"]=="Pedestrian":
                pcd_colors[inlier_indices] = [0,1,0]
            elif label["label"]=="Vehicle":
                pcd_colors[inlier_indices] = [0,0,1]
            else:
                pcd_colors[inlier_indices] = [1,1,0]



            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(bbx)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            line_sets.append(line_set)
        o3d.visualization.draw_geometries([pcd,*line_sets])
        # o3d.visualization.draw_geometries([pcd])

    def project_velo_to_ref(self,pts_3d_velo):
        #add 1 to turn n*3 into n*4
        pts_3d_velo = np.c_[pts_3d_velo,np.ones(len(pts_3d_velo))]  
        return np.matmul(pts_3d_velo,self.V2C.T) 

    def project_ref_to_velo(self,pts_3d_ref):
        pts_3d_ref = np.c_[pts_3d_ref,np.ones(len(pts_3d_ref))]  
        return np.matmul(pts_3d_ref,self.C2V.T)

    def project_rect_to_ref(self,pts_3d_rect):
        #input n*3, output n*3
        return (np.matmul(np.linalg.inv(self.R0),pts_3d_rect.T)).T

    def project_ref_to_rect(self,pts_3d_ref):
        #input n*3, output n*3
        return (np.matmul(self.R0,pts_3d_ref.T)).T

    def project_velo_to_rect(self,pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    def project_rect_to_velo(self,pts_3d_rect):
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_rect_to_image(self,pts_3d_rect):
        """
            inputs: n*3 points in rect camera coordinate
            outputs: n*2 points in image2 coordinate
        """
        pts_3d_rect = np.c_[pts_3d_rect,np.ones(len(pts_3d_rect))]  
        #n*4*(4*3)=n*3
        pts_2d = np.matmul(pts_3d_rect,self.P.T)
        pts_2d[:,0] = pts_2d[:,0]/pts_2d[:,2]
        pts_2d[:,1] = pts_2d[:,1]/pts_2d[:,2]
        return pts_2d[:,0:2]

    def project_velo_to_image(self,pts_3d_velo):
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        # pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts_3d_rect))
        # pcd.paint_uniform_color([0, 0, 0])
        # o3d.visualization.draw_geometries([pcd])
        return self.project_rect_to_image(pts_3d_rect)

    def get_bbx(self,label):
        """get the bounding box in velo coordinate
        """
        label2 = np.zeros(len(label))
        label2[1:] = np.array([float(x) for x in label[1:]])
        h,w,l = label2[8],label2[9],label2[10]
        x_corners = [l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2,l/2]
        y_corners = [0,0,0,0,-h,-h,-h,-h]
        z_corners = [w/2,w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2]
        box_corner = np.c_[x_corners,y_corners,z_corners]
        #rotate in y axis theta degree
        theta = np.float(label2[14])
        rotatey = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        rotate_box = np.matmul(rotatey,box_corner.T).T
        location = np.array([label2[11],label2[12],label2[13]])
        #calculate the exact position of the box
        rotate_box[:,0] = rotate_box[:,0] + location[0]
        rotate_box[:,1] = rotate_box[:,1] + location[1]
        rotate_box[:,2] = rotate_box[:,2] + location[2]
        bbx = self.project_rect_to_velo(rotate_box)
        return bbx[:,:3]

    def in_hull(self,p, hull):
        """
        Test if points in `p` are in `hull`
        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed https://blog.csdn.net/qq_35632833/article/details/106865648
        """
        from scipy.spatial import Delaunay
        if not isinstance(hull,Delaunay):
            hull = Delaunay(hull)

        return hull.find_simplex(p)>=0


if __name__=="__main__":
    kitti = Kittidata("/home/shirleyzzr/dataset/kitti/object","training",10)
    kitti.show_lidar_with_boxes()





