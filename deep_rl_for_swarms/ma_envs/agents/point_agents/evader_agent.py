import scipy.spatial as ssp
import numpy as np
import deep_rl_for_swarms.ma_envs.commons.utils as U
import shapely.geometry as sg
from deep_rl_for_swarms.ma_envs.base import Agent


class Evader(Agent):
    def __init__(self, experiment):
        super(Evader, self).__init__()
        self.obs_radius = experiment.obs_radius
        self.world_size = experiment.world_size
        self.torus = experiment.torus
        self.dynamics = 'direct'
        self.max_speed = 2 * 10  # cm/s
        if self.torus:
            self.bounding_box = np.array([0., 2 * self.world_size, 0., 2 * self.world_size])
        else:
            self.bounding_box = np.array([0., self.world_size, 0., self.world_size])

        self.action_callback = self.step

    def reset(self, state):
        self.state.p_pos = state
        self.state.p_vel = np.zeros(2)

    def step(self, agent, world):
        if self.torus:
            points_center = np.vstack([world.agent_states[:, 0:2], self.state.p_pos])
            pursuers_down_right = np.hstack([world.agent_states[:, 0:1] + world.world_size, world.agent_states[:, 1:2]])
            pursuers_up_left = np.hstack([world.agent_states[:, 0:1], world.agent_states[:, 1:2] + world.world_size])
            pursuers_up_right = np.hstack(
                [world.agent_states[:, 0:1] + world.world_size, world.agent_states[:, 1:2] + world.world_size])
            evader_down_right = np.hstack([self.state.p_pos[0:1] + world.world_size, self.state.p_pos[1:2]])
            evader_up_left = np.hstack([self.state.p_pos[0:1], self.state.p_pos[1:2] + world.world_size])
            evader_up_right = np.hstack([self.state.p_pos[0:1] + world.world_size, self.state.p_pos[1:2] + world.world_size])
            points_down_right = np.hstack([points_center[:, 0:1] + world.world_size, points_center[:, 1:2]])
            points_up_left = np.hstack([points_center[:, 0:1], points_center[:, 1:2] + world.world_size])
            points_up_right = np.hstack(
                [points_center[:, 0:1] + world.world_size, points_center[:, 1:2] + world.world_size])

            nodes = np.vstack([world.agent_states[:, 0:2],
                               pursuers_down_right,
                               pursuers_up_left,
                               pursuers_up_right,
                               self.state.p_pos,
                               evader_down_right,
                               evader_up_left,
                               evader_up_right])

            dist_matrix_full = U.get_euclid_distances(nodes)

            quadrant_check = np.sign(self.state.p_pos - world.world_size / 2)
            if np.all(quadrant_check == np.array([1, 1])):
                evader_quadrant = 0
            elif np.all(quadrant_check == np.array([-1, 1])):
                evader_quadrant = 1
            elif np.all(quadrant_check == np.array([1, -1])):
                evader_quadrant = 2
            elif np.all(quadrant_check == np.array([-1, -1])):
                evader_quadrant = 3

            evader_dist = dist_matrix_full[:-4, -4 + evader_quadrant]
            sub_list = list(np.where(evader_dist < self.obs_radius)[0])
            if len(sub_list) > 10:
                sub_list = list(np.argsort(evader_dist)[0:10])
            sub_list.append(4 * world.nr_agents + evader_quadrant)
            evader_sub = len(sub_list) - 1
            closest_pursuer = np.where(evader_dist == evader_dist.min())[0]

            nodes_center_sub = nodes[sub_list, :]
            nodes_left = np.copy(nodes_center_sub)
            nodes_left[:, 0] = self.bounding_box[0] - (nodes_left[:, 0] - self.bounding_box[0])
            nodes_right = np.copy(nodes_center_sub)
            nodes_right[:, 0] = self.bounding_box[1] + (self.bounding_box[1] - nodes_right[:, 0])
            nodes_down = np.copy(nodes_center_sub)
            nodes_down[:, 1] = self.bounding_box[2] - (nodes_down[:, 1] - self.bounding_box[2])
            nodes_up = np.copy(nodes_center_sub)
            nodes_up[:, 1] = self.bounding_box[3] + (self.bounding_box[3] - nodes_up[:, 1])

            points = np.vstack([nodes_center_sub, nodes_down, nodes_left, nodes_right, nodes_up])

        else:
            nodes = np.vstack([world.agent_states[:, 0:2],
                               self.state.p_pos,
                               ])
            distances = U.get_euclid_distances(nodes)
            evader_dist = distances[-1, :-1]
            closest_pursuer = np.where(evader_dist == evader_dist.min())[0]
            sub_list = list(np.where(evader_dist < self.obs_radius)[0])
            if len(sub_list) > 10:
                sub_list = list(np.argsort(evader_dist)[0:10])
            sub_list.append(world.nr_agents)
            evader_sub = len(sub_list) - 1

            nodes_center_sub = nodes[sub_list, :]
            nodes_left = np.copy(nodes_center_sub)
            nodes_left[:, 0] = self.bounding_box[0] - (nodes_left[:, 0] - self.bounding_box[0])
            nodes_right = np.copy(nodes_center_sub)
            nodes_right[:, 0] = self.bounding_box[1] + (self.bounding_box[1] - nodes_right[:, 0])
            nodes_down = np.copy(nodes_center_sub)
            nodes_down[:, 1] = self.bounding_box[2] - (nodes_down[:, 1] - self.bounding_box[2])
            nodes_up = np.copy(nodes_center_sub)
            nodes_up[:, 1] = self.bounding_box[3] + (self.bounding_box[3] - nodes_up[:, 1])

            points = np.vstack([nodes_center_sub, nodes_down, nodes_left, nodes_right, nodes_up])

        vor = ssp.Voronoi(points)

        d = np.zeros(2)

        for i, ridge in enumerate(vor.ridge_points):
            if evader_sub in set(ridge) and np.all([r <= evader_sub for r in ridge]):
                if self.torus:
                    neighbor = min([sub_list[r] for r in ridge])
                else:
                    # neighbor = min(ridge)
                    neighbor = min([sub_list[r] for r in ridge])

                if neighbor in closest_pursuer:
                    ridge_inds = vor.ridge_vertices[i]
                    a = vor.vertices[ridge_inds[0], :]
                    b = vor.vertices[ridge_inds[1], :]

                    line_of_control = b - a
                    L_i = np.linalg.norm(line_of_control)

                    if self.torus:
                        xi = nodes[neighbor, :] - nodes[4 * world.nr_agents + evader_quadrant]
                    else:
                        xi = nodes[neighbor, :] - self.state.p_pos
                    eta_h_i = xi / np.linalg.norm(xi)
                    eta_v_i = np.array([-eta_h_i[1], eta_h_i[0]])

                    if self.torus:
                        line1 = sg.LineString([nodes[4 * world.nr_agents + evader_quadrant], nodes[neighbor, :]])
                    else:
                        line1 = sg.LineString([self.state.p_pos, nodes[neighbor, :]])
                    line2 = sg.LineString([a, b])
                    intersection = line1.intersection(line2)

                    if not intersection.is_empty:
                        inter_point = np.hstack(intersection.xy)

                        if np.dot(line_of_control, eta_v_i.flatten()) > 0:
                            l_i = np.linalg.norm(a - inter_point)
                        else:
                            l_i = np.linalg.norm(b - inter_point)
                    else:
                        if np.dot(line_of_control, eta_v_i.flatten()) > 0:
                            l_i = 0
                        else:
                            l_i = L_i

                    alpha_h_i = - L_i / 2
                    alpha_v_i = (l_i ** 2 - (L_i - l_i) ** 2) / (2 * np.linalg.norm(xi))

                    d = (alpha_h_i * eta_h_i - alpha_v_i * eta_v_i) / np.sqrt(alpha_h_i ** 2 + alpha_v_i ** 2)

        assert ('d' in locals())

        return d
