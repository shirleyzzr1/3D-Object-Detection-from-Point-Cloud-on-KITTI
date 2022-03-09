import numpy as np
import octreelite as octree
from result_set import RadiusNNResultSet
class DBSCAN():
    def __init__(self):
        self.visited = []
    def fit(self,data):
        leaf_size = 10
        min_extent = 0.001
        radius = 0.5
        minsamples = 5
        cluster_num = 1
        self.visited = np.zeros(len(data))
        root = octree.octree_construction(data, leaf_size, min_extent)

        for i in range(len(data)):
            if self.visited[i]!=0:
                continue
            result_set = []
            octree.octree_radius_search_fast(root, data, result_set, np.asarray(data[i,:]),radius)
            if len(result_set) <= minsamples:            #noise point
                self.visited[i]=-1
            else:
                self.visited[i] = cluster_num        #core point
                neighbours = result_set                  #neightbour points
                while len(neighbours) != 0:
                    index = neighbours.pop()
                    if self.visited[index]!=0:
                        continue
                    # searched.append(index)
                    # print("index", index)
                    result_set = []
                    octree.octree_radius_search_fast(root, data, result_set, np.asarray(data[index,:]),radius)
                    if len(result_set) >= minsamples:
                        self.visited[index] = cluster_num
                        neighbours.extend(result_set)
                    else:
                        self.visited[index] = cluster_num       
                cluster_num+=1
        print(self.visited)
    def predict(self,data):
        return self.visited