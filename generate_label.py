"""
turn all groundtruth labels "Car,Van,Truck,Tram" into "Vehicle"
"""
import numpy as np
import os
if __name__=="__main__":
    kitti_dir = "../../../dataset/kitti/object"
    split_dir = os.path.join(kitti_dir,"ImageSets","val"+".txt")
    split_name = np.loadtxt(split_dir)
    label_dir = os.path.join(kitti_dir,"training","label_2")
    for idx in split_name:
        label_filename = os.path.join(label_dir,"%06d.txt" % (idx))
        lines = []
        with open(label_filename, "r") as f:
            for line in f.readlines():
                linesets = line.split(" ")
                if linesets[0] in ["Car","Van","Truck","Tram"]:
                    linesets[0] = "Vehicle"
                linestr = " ".join(linesets)
                lines.append(linestr)
        new_label_filename = os.path.join("./gt_label","%06d"%int(idx)+".txt")
        with open(new_label_filename,"a") as f:
            for line in lines:
                f.write(line)
