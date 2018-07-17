import numpy as np
import scipy.spatial as ssp
import scipy.stats as sst


class EzPickle(object):
    """Objects that are pickled and unpickled via their constructor
    arguments.

    Example usage:

        class Dog(Animal, EzPickle):
            def __init__(self, furcolor, tailkind="bushy"):
                Animal.__init__()
                EzPickle.__init__(furcolor, tailkind)
                ...

    When this object is unpickled, a new Dog will be constructed by passing the provided
    furcolor and tailkind into the constructor. However, philosophers are still not sure
    whether it is still the same dog.

    """

    def __init__(self, *args, **kwargs):
        self._ezpickle_args = args
        self._ezpickle_kwargs = kwargs

    def __getstate__(self):
        return {"_ezpickle_args": self._ezpickle_args, "_ezpickle_kwargs": self._ezpickle_kwargs}

    def __setstate__(self, d):
        out = type(self)(*d["_ezpickle_args"], **d["_ezpickle_kwargs"])
        self.__dict__.update(out.__dict__)


def get_angle(x0, x1, torus=False, world_size=None, positive=False):
    delta = x0 - x1
    if torus:
        delta = np.where(delta > world_size / 2, delta - world_size, delta)
        delta = np.where(delta < -world_size / 2, delta + world_size, delta)
    angle = np.arctan2(delta[:, 1], delta[:, 0])

    if positive:
        angle = np.where(angle < 0, angle + 2 * np.pi, angle)

    return angle


def get_distance_matrix(points, world_size=None, torus=False, add_to_diagonal=0):
    distance_matrix = np.vstack([get_distances(points, p, torus=torus, world_size=world_size) for p in points])
    distance_matrix = distance_matrix + np.diag(add_to_diagonal * np.ones(points.shape[0]))
    return distance_matrix


def get_upper_triangle(matrix, subtract_from_diagonal=0):
    matrix = matrix - np.diag(subtract_from_diagonal * np.ones(matrix.shape[0]))
    triangle = ssp.distance.squareform(matrix)
    return triangle


def get_distances(x0, x1, torus=False, world_size=None):
    delta = np.abs(x0 - x1)
    if torus:
        delta = np.where(delta > world_size / 2, delta - world_size, delta)
    dist = np.sqrt((delta ** 2).sum(axis=-1))
    return dist


def get_euclid_distances(points, matrix=True):
    if matrix:
        dist = ssp.distance.squareform(
            ssp.distance.pdist(points, 'euclidean'))
    else:
        dist = ssp.distance.pdist(points, 'euclidean')
    return dist


def get_adjacency_matrix(distance_matrix, max_dist):
    return np.array((distance_matrix < max_dist) & (distance_matrix > 0.), dtype=float)


def dfs(adj_matrix, minsize):
    """ depth first search

    Returns subsets with at least minsize or more nodes.
    """

    visited = set()
    connected_sets = []

    for ind, row in enumerate(adj_matrix):
        if ind not in visited:
            connected_sets.append(set())
            stack = [ind]
            while stack:
                vertex = stack.pop()
                if vertex not in visited:
                    visited.add(vertex)
                    connected_sets[-1].add(vertex)
                    stack.extend(set(np.where(adj_matrix[vertex, :] != 0)[0]) - visited)
    return [cs for cs in connected_sets if len(cs) >= minsize]


def get_connected_sets(sets):
    # find unique directly connected
    # test_sets = [set(np.array(list(s)) % self.nr_actors) for s in sets]
    final_sets = []

    for i, s in enumerate(sets):
        if final_sets:
            if s not in [fs[1] for fs in final_sets]:
                is_super_set = [s >= fs[1] for fs in final_sets]
                if any(is_super_set):
                    del final_sets[np.where(is_super_set)[0][0]]
                is_sub_set = [s <= fs[1] for fs in final_sets]
                if not any(is_sub_set):
                    final_sets.append([i, s])
        else:
            final_sets.append([i, s])

    indices = [fs[0] for fs in final_sets]

    setlist = [list(sets[i]) for i in indices]

    return setlist


def get_basis_fct(radius, nr_basis_fct, scale=0.05):
    x = np.reshape(np.linspace(0, radius, radius * 100 + 1), [-1, 1])
    mu = np.reshape(np.linspace(0, radius, nr_basis_fct), [1, nr_basis_fct])
    basis_fct = sst.norm.pdf(x, loc=mu, scale=scale * np.ones([1, nr_basis_fct]))
    return basis_fct


def get_basis_fct_2d(radius, nr_dist_basis_fct, nr_bear_basis_fct, scale=(0.05, 0.05)):
    mu_x = np.linspace(0, radius, nr_dist_basis_fct)
    mu_y = np.linspace(0, 2 * np.pi, nr_bear_basis_fct)
    xv, yv = np.meshgrid(mu_x, mu_y)
    mu = np.stack([xv.flatten(), yv.flatten()], axis=1)
    x = np.linspace(0, radius, radius * 5)
    y = np.linspace(0, 2 * np.pi, 315)

    xx, yy = np.meshgrid(x, y)
    xy = np.stack([xx.flatten(), yy.flatten()], axis=1)
    basis_fct = []
    for m in mu:
        basis_fct.append(sst.multivariate_normal.pdf(xy, m, np.diag(scale)))

    out = np.stack(basis_fct, axis=0).reshape(nr_dist_basis_fct * nr_bear_basis_fct, y.size, x.size)
    return out


def get_weights_2d(points, mu, s, bins):
    x = np.reshape(points, [-1, 2])

    ww = 1 / (2 * np.pi * s[0] * s[1]) * np.exp(-1/2 * ((x[:, 0] - mu[:, 0][:, None])**2 / (s[0]**2) +
                                                        (x[:, 1] - mu[:, 1][:, None])**2 / (s[1]**2)))
    # TODO: find better way of normalisation
    www = ww / np.sum(ww, axis=0)
    weights = np.sum(www, axis=1)
    return weights.reshape([bins[0], bins[1]])


def get_weights_3d(points, mu, s, bins):
    x = np.reshape(points, [-1, 3])

    ww = np.exp(-1/2 * ((x[:, 0] - mu[:, 0][:, None]) ** 2 / (s[0] ** 2) +
                        (x[:, 1] - mu[:, 1][:, None]) ** 2 / (s[1] ** 2) +
                        (x[:, 2] - mu[:, 2][:, None]) ** 2 / (s[2] ** 2)))

    www = ww / np.sum(ww, axis=0)
    weights = np.sum(www, axis=1)
    return weights


def unicycle_to_single_integrator(dxu, poses, projection_distance=0.05):
    """A function for converting from unicycle to single-integrator dynamics.
    Utilizes a virtual point placed in front of the unicycle.

    dxu: 2xN numpy array of unicycle control inputs
    poses: 3xN numpy array of unicycle poses
    projection_distance: How far ahead of the unicycle model to place the point

    -> 2xN numpy array of single-integrator control inputs
    """

    N, M = np.shape(dxu)

    cs = np.cos(poses[:, 2])
    ss = np.sin(poses[:, 2])

    dxi = np.zeros((N, 2))
    dxi[:, 0] = (cs*dxu[:, 0] - projection_distance*ss*dxu[:, 1])
    dxi[:, 1] = (ss*dxu[:, 0] + projection_distance*cs*dxu[:, 1])

    return dxi


wheel_radius = 1.5
robot_radius = 3.5


def forward_kinematics(w_l, w_r):
    c_l = wheel_radius * w_l
    c_r = wheel_radius * w_r
    v = (c_l + c_r) / 2
    a = (c_r - c_l) / (2 * robot_radius)
    return (v, a)


# computing the inverse kinematics for a differential drive
def inverse_kinematics(v, a):
    c_l = v - (robot_radius * a)
    c_r = v + (robot_radius * a)
    w_l = c_l / wheel_radius
    w_r = c_r / wheel_radius
    return (w_l, w_r)
