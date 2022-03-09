import numpy as np
class Octant:
    def __init__(self, children, center, extent, point_indices, is_leaf):
        self.children = children
        self.center = center
        self.extent = extent
        self.point_indices = point_indices
        self.is_leaf = is_leaf

    def __str__(self):
        output = ''
        output += 'center: [%.2f, %.2f, %.2f], ' % (self.center[0], self.center[1], self.center[2])
        output += 'extent: %.2f, ' % self.extent
        output += 'is_leaf: %d, ' % self.is_leaf
        output += 'children: ' + str([x is not None for x in self.children]) + ", "
        output += 'point_indices: ' + str(self.point_indices)
        return output


def octree_recursive_build(root, db, center, extent, point_indices, leaf_size, min_extent):
    if len(point_indices) == 0:
        return None

    if root is None:
        root = Octant([None for i in range(8)], center, extent, point_indices, is_leaf=True)

    # determine whether to split this octant
    if len(point_indices) <= leaf_size or extent <= min_extent:
        root.is_leaf = True
    else:

        root.is_leaf=False
        children_point_indeces = [[]for i in range(8)]

        for point_idx in point_indices:
            point_db = db[point_idx]
            morton_code = 0
            if point_db[0]> center[0]:
                morton_code = morton_code | 1
            if point_db[1] > center[1]:
                morton_code = morton_code | 2
            if point_db[2] > center[2]:
                morton_code = morton_code | 4
            children_point_indeces[morton_code].append(point_idx)
        factor = [-0.5,0.5]
        for i in range(8):
            child_center_x = center[0] + factor[(i&1)>0] *extent #1，3，5，7 equals0.5
            child_center_y = center[1] + factor[(i & 2) > 0] * extent#2~3，6~7 equals 0.5
            child_center_z = center[2] + factor[(i & 4) > 0] * extent#4~7equals 0.5，0~3 equals -0.5
            child_extent = 0.5*extent
            child_center = np.asarray([child_center_x,child_center_y,child_center_z])
            root.children[i]=octree_recursive_build(root.children[i],db,child_center,child_extent,
                                                    children_point_indeces[i],leaf_size,min_extent)

    return root

def octree_construction(db_np, leaf_size, min_extent):
    N, dim = db_np.shape[0], db_np.shape[1]
    db_np_min = np.amin(db_np, axis=0)
    db_np_max = np.amax(db_np, axis=0)
    db_extent = np.max(db_np_max - db_np_min) * 0.5
    db_center = np.mean(db_np, axis=0)

    root = None
    root = octree_recursive_build(root, db_np, db_center, db_extent, list(range(N)),
                                  leaf_size, min_extent)

    return root
def inside(query: np.ndarray, radius: float, octant:Octant):
    """
    Determines if the query ball is inside the octant
    :param query:
    :param radius:
    :param octant:
    :return:
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)
    possible_space = query_offset_abs + radius
    return np.all(possible_space < octant.extent)

def overlaps(query: np.ndarray, radius: float, octant:Octant):
    """
    Determines if the query ball overlaps with the octant
    :param query:
    :param radius:
    :param octant:
    :return:
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)

    # completely outside, since query is outside the relevant area
    max_dist = radius + octant.extent
    if np.any(query_offset_abs > max_dist):
        return False

    # if pass the above check, consider the case that the ball is contacting the face of the octant
    if np.sum((query_offset_abs < octant.extent).astype(np.int)) >= 2:
        return True

    # conside the case that the ball is contacting the edge or corner of the octant
    # since the case of the ball center (query) inside octant has been considered,
    # we only consider the ball center (query) outside octant
    x_diff = max(query_offset_abs[0] - octant.extent, 0)
    y_diff = max(query_offset_abs[1] - octant.extent, 0)
    z_diff = max(query_offset_abs[2] - octant.extent, 0)

    return x_diff * x_diff + y_diff * y_diff + z_diff * z_diff < radius * radius
def octree_radius_search_fast(root: Octant, db: np.ndarray, result_set, query: np.ndarray,radius):
    if root is None:
        return False

    if root.is_leaf and len(root.point_indices) > 0:
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            if diff[i] >radius:
                return
            result_set.append(root.point_indices[i])
        # check whether we can stop search now
        return inside(query, radius, root)

    # no need to go to most relevant child first, because anyway we will go through all children
    for c, child in enumerate(root.children):
        if child is None:
            continue
        if False == overlaps(query, radius, child):
            continue
        if octree_radius_search_fast(child, db, result_set, query,radius):
            return True

    return inside(query, radius, root)
