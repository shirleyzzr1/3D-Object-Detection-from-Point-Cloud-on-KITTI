#http://www.cppblog.com/lovedday/archive/2008/02/23/43122.html
#adopt from the C++ version to python version
"""
  //--------------------------------------------------------------------------------------
    // Return one of the 8 corner points.  The points are numbered as follows:
    //
    //            7                                8
    //              ------------------------------
    //             /|                           /|
    //            / |                          / |
    //           /  |                         /  |
    //          /   |                        /   |
    //         /    |                       /    |
    //        /     |                      /     |
    //       /      |                     /      |
    //      /       |                    /       |
    //     /        |                   /        |
    //  6 /         |                5 /         |
    //   /----------------------------/          |
    //   |          |                 |          |
    //   |          |                 |          |      -Y
    //   |        3 |                 |          | 
    //   |          |-----------------|----------|      |
    //   |         /                  |         /  4    |
    //   |        /                   |        /        |       -x
    //   |       /                    |       /         |
    //   |      /                     |      /          |     /
    //   |     /                      |     /           |    /
    //   |    /                       |    /            |   /
    //   |   /                        |   /             |  /
    //   |  /                         |  /              | /
    //   | /                          | /               |/
    //   |/                           |/                ----------------- +Z
    //   ------------------------------
    //  2                              1
    //
"""
import numpy as np
class BoundingBox():
    def __init__(self) -> None:
        self.min = []
        self.max = []
        self.empty()
    def x_size(self):
        self.max[0] - self.min[0]
    def y_size(self):
        self.max[1] - self.min[1]
    def z_size(self):
        self.max[2] - self.min[2]
    def center(self):
        return (np.array(self.min) + np.array(self.max))/2

    #empty the current box
    def empty(self):
        self.min = np.ones(3)*float("inf")
        self.max = np.ones(3)*float("-inf")
    
    #add a point to the box
    def add(self,p):
        if(p[0]<self.min[0]):self.min[0] = p[0]
        if(p[0]>self.max[0]):self.max[0] = p[0]
        if(p[1]<self.min[1]):self.min[1] = p[1]
        if(p[1]>self.max[1]):self.max[1] = p[1]        
        if(p[2]<self.min[2]):self.min[2] = p[2]
        if(p[2]>self.max[2]):self.max[2] = p[2]


    def is_empty(self):
        #check if on any axis is inverted
        return (self.min[0]>self.max[0]) \
                and (self.min[1]>self.max[1])\
                and(self.min[2]>self.max[2]) 
        
    #check is the box contains a certain point
    def contains(self,p):
        px,py,pz = p
        minx,miny,minz = self.min
        maxx,maxy,maxz = self.max
        return (px>=minx) and (px<=maxx) and(py>=miny) and (py<=maxy)\
                and (pz>=minz) and (pz<=maxz)
    #
    #0001
    #0010
    #0100
    def get_box(self,points):
        for i in range(len(points)):
            self.add(points[i])
        h = self.max[1] - self.min[1]
        w = self.max[2] - self.min[2]
        l = self.max[0] - self.min[0]
        center_xyz = self.center()
        center_xyz[1] += h/2
        # ry = np.arctan2(self.max[0],self.max[2])
        ry = 0

        return h,w,l,center_xyz[0],center_xyz[1],center_xyz[2],ry
    
