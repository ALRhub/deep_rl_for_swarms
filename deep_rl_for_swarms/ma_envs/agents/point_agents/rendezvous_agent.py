from deep_rl_for_swarms.ma_envs.base import Agent
import deep_rl_for_swarms.ma_envs.commons.utils as U
from gym import spaces
import numpy as np
import fast_histogram as fh


class PointAgent(Agent):
    def __init__(self, experiment):
        super(PointAgent, self).__init__()
        self.comm_radius = experiment.comm_radius
        self.obs_radius = experiment.comm_radius / 2
        self.obs_mode = experiment.obs_mode
        self.distance_bins = experiment.distance_bins
        self.bearing_bins = experiment.bearing_bins
        self.torus = experiment.torus
        self.n_agents = experiment.nr_agents
        self.world_size = experiment.world_size
        self._dim_a = 2
        self.dim_local_o = 3 + int(not self.torus)

        if self.obs_mode == '2d_rbf_acc':
            mu_d = np.linspace(0, self.world_size * np.sqrt(2), self.distance_bins)
            mu_b = np.linspace(0, 2 * np.pi, self.bearing_bins)
            s_d = 4 * self.comm_radius / 80
            s_b = 0.33
            xv, yv = np.meshgrid(mu_d, mu_b)
            self.mu = np.stack([xv.flatten(), yv.flatten()], axis=1)
            self.s = np.hstack([s_d, s_b])
            self.dim_local_o = 2 + 3 * int(not self.torus)
            self.dim_rec_o = (self.distance_bins, self.bearing_bins)
            self.dim_flat_o = self.dim_local_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
            self.dim_mean_embs = None
        elif self.obs_mode == '3d_rbf':
            mu_d = np.linspace(0, self.world_size * np.sqrt(2), self.distance_bins)
            mu_b = np.linspace(0, 2 * np.pi, self.bearing_bins)
            s_d = 4 * self.comm_radius / 80
            s_b = 0.33
            xv, yv, zv = np.meshgrid(mu_d, mu_b, mu_b)
            self.mu = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=1)
            self.s = np.hstack([s_d, s_b, s_b])
            self.dim_local_o = 2 + 3 * int(not self.torus)
            self.dim_rec_o = (self.distance_bins, self.bearing_bins, self.bearing_bins)
            self.dim_flat_o = self.dim_local_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
            self.dim_mean_embs = None
        elif self.obs_mode == '2d_rbf_acc_limited':
            mu_d = np.linspace(0, self.comm_radius, self.distance_bins)
            mu_b = np.linspace(0, 2 * np.pi, self.bearing_bins)
            s_d = 4 * self.comm_radius / 80
            s_b = 0.33
            xv, yv = np.meshgrid(mu_d, mu_b)
            self.mu = np.stack([xv.flatten(), yv.flatten()], axis=1)
            self.s = np.hstack([s_d, s_b])
            self.dim_local_o = 3 + 3 * int(not self.torus)
            self.dim_rec_o = (self.distance_bins, self.bearing_bins)
            self.dim_flat_o = self.dim_local_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
            self.dim_mean_embs = None
        elif self.obs_mode == '2d_rbf_limited':
            mu_d = np.linspace(0, self.comm_radius, self.distance_bins)
            mu_b = np.linspace(0, 2 * np.pi, self.bearing_bins)
            s_d = 4 * self.comm_radius / 80
            s_b = 0.33
            xv, yv = np.meshgrid(mu_d, mu_b)
            self.mu = np.stack([xv.flatten(), yv.flatten()], axis=1)
            self.s = np.hstack([s_d, s_b])
            self.dim_local_o = 1 + 3 * int(not self.torus)
            self.dim_rec_o = (self.distance_bins, self.bearing_bins)
            self.dim_flat_o = self.dim_local_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
            self.dim_mean_embs = None
        elif self.obs_mode == '2d_hist_acc':
            self.dim_rec_o = (self.bearing_bins, self.distance_bins)
            self.dim_local_o = 2 + 3 * int(not self.torus)
            self.dim_flat_o = self.dim_local_o
            self.dim_mean_embs = None
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
        elif self.obs_mode == 'sum_obs_acc':
            self.dim_rec_o = (self.n_agents - 1, 7)
            self.dim_mean_embs = self.dim_rec_o
            self.dim_local_o = 2 + 3 * int(not self.torus)
            self.dim_flat_o = self.dim_local_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
        elif self.obs_mode == 'sum_obs_acc_full':
            self.dim_rec_o = (100, 9)
            self.dim_mean_embs = self.dim_rec_o
            self.dim_local_o = 2 + 3 * int(not self.torus)
            self.dim_flat_o = self.dim_local_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
        elif self.obs_mode == 'sum_obs_acc_no_vel':
            self.dim_rec_o = (100, 5)
            self.dim_mean_embs = self.dim_rec_o
            self.dim_local_o = 2 + 3 * int(not self.torus)
            self.dim_flat_o = self.dim_local_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
        elif self.obs_mode == 'sum_obs_acc_limited':
            self.dim_rec_o = (100, 8)
            self.dim_mean_embs = self.dim_rec_o
            self.dim_local_o = 3 + 3 * int(not self.torus)
            self.dim_flat_o = self.dim_local_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
        elif self.obs_mode == 'sum_obs':
            self.dim_rec_o = (100, 7)
            self.dim_mean_embs = self.dim_rec_o
            self.dim_local_o = 1 + 3 * int(not self.torus)
            self.dim_flat_o = self.dim_local_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
        elif self.obs_mode == 'sum_obs_limited':
            self.dim_rec_o = (100, 8)
            self.dim_mean_embs = self.dim_rec_o
            self.dim_local_o = 1 + 3 * int(not self.torus)
            self.dim_flat_o = self.dim_local_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
        elif self.obs_mode == 'fix_acc':
            self.dim_rec_o = (self.n_agents - 1, 5)
            self.dim_local_o = 2 + 3 * int(not self.torus)
            self.dim_flat_o = self.dim_local_o
            self.dim_mean_embs = None
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_local_o

        else:
            raise ValueError('obs mode must be 1D or 2D')
        self.r_matrix = None
        self.feature = None
        self.complete = False
        self.n_sensors = 4
        self.sensor_range = 0.5
        self.radius = 0.2
        angles_K = np.linspace(0., 2. * np.pi, self.n_sensors + 1)[:-1]
        sensor_vecs_K_2 = np.c_[np.cos(angles_K), np.sin(angles_K)]

        self.sensors = sensor_vecs_K_2
        self.rel_vel_hist = []
        self.neighborhood_size_hist = []

    @property
    def observation_space(self):
        ob_space = spaces.Box(low=0., high=1., shape=(self._dim_o,), dtype=np.float32)
        ob_space.dim_local_o = self.dim_local_o
        ob_space.dim_flat_o = self.dim_flat_o
        ob_space.dim_rec_o = self.dim_rec_o
        ob_space.dim_mean_embs = self.dim_mean_embs
        return ob_space

    @property
    def action_space(self):
        return spaces.Box(np.array([-1., -1.]), np.array([1., 1.]), dtype=np.float32)

    def set_velocity(self, vel):
        self.velocity = vel

    def reset(self, state):
        self.state.p_pos = state[0:2]
        self.state.p_orientation = state[2]
        self.state.p_vel = np.zeros(2)
        self.state.w_vel = np.zeros(2)
        self.feature = np.inf
        self.complete = False

    def get_observation(self, dm, my_orientation, their_orientation, vels, nh_size):

        if self.obs_mode == 'fix_acc':
            ind = np.where(dm == -1)[0][0]
            rel_vels = self.state.w_vel - vels

            local_obs = self.get_local_obs_acc()

            fix_obs = np.zeros(self.dim_rec_o)

            fix_obs[:, 0] = np.concatenate([dm[0:ind], dm[ind + 1:]]) / self.comm_radius
            fix_obs[:, 1] = np.cos(np.concatenate([my_orientation[0:ind], my_orientation[ind + 1:]]))
            fix_obs[:, 2] = np.sin(np.concatenate([my_orientation[0:ind], my_orientation[ind + 1:]]))
            fix_obs[:, 3] = np.concatenate([rel_vels[0:ind, 0], rel_vels[ind + 1:, 0]]) / (2 * self.max_lin_velocity)
            fix_obs[:, 4] = np.concatenate([rel_vels[0:ind, 1], rel_vels[ind + 1:, 1]]) / (2 * self.max_lin_velocity)

            obs = np.hstack([fix_obs.flatten(), local_obs.flatten()])

        elif self.obs_mode == '2d_hist_acc':
            local_obs = self.get_local_obs_acc()
            in_range = (0 < dm) & (dm < self.comm_radius)
            hist_2d = fh.histogram2d(my_orientation[in_range], dm[in_range],
                                     bins=(self.bearing_bins, self.distance_bins),
                                     range=[[-np.pi, np.pi], [0, self.world_size * np.sqrt(2)]])
            histogram = hist_2d.flatten() / (self.n_agents - 1)
            obs = np.hstack([histogram, local_obs])

        elif self.obs_mode == '2d_rbf_acc':
            in_range = (dm < self.comm_radius) & (0 < dm)

            local_obs = self.get_local_obs_acc()

            if np.any(in_range):
                dbn = np.stack([dm[in_range], my_orientation[in_range] + np.pi], axis=1)
                rbf_hist = U.get_weights_2d(dbn, self.mu, self.s,
                                            [self.bearing_bins, self.distance_bins]) / (self.n_agents - 1)

            else:
                rbf_hist = np.zeros([self.bearing_bins, self.distance_bins])

            rbf_hist_flat = rbf_hist

            obs = np.hstack([rbf_hist_flat, local_obs])

        elif self.obs_mode == '3d_rbf':
            in_range = (dm < self.comm_radius) & (0 < dm)

            local_obs = self.get_local_obs_acc()

            if np.any(in_range):
                dbn = np.stack([dm[in_range],
                                my_orientation[in_range] + np.pi,
                                their_orientation[in_range] + np.pi],
                               axis=1)
                rbf_hist = U.get_weights_3d(dbn, self.mu, self.s,
                                            [self.bearing_bins, self.distance_bins, self.bearing_bins]) / (self.n_agents - 1)

            else:
                rbf_hist = np.zeros([self.bearing_bins, self.distance_bins, self.bearing_bins])

            rbf_hist_flat = rbf_hist.flatten()

            obs = np.hstack([rbf_hist_flat, local_obs])

        elif self.obs_mode == '2d_rbf_acc_limited':
            in_range = (dm < self.comm_radius) & (0 < dm)
            nr_neighbors = np.sum(in_range)

            local_obs = self.get_local_obs_acc()
            local_obs[-1] = nr_neighbors / (self.n_agents - 1)

            if np.any(in_range):
                dbn = np.stack([dm[in_range], my_orientation[in_range] + np.pi], axis=1)
                rbf_hist = U.get_weights_2d(dbn, self.mu, self.s,
                                            [self.bearing_bins, self.distance_bins]) / (self.n_agents - 1)

            else:
                rbf_hist = np.zeros([self.bearing_bins, self.distance_bins])

            rbf_hist_flat = rbf_hist.flatten()

            obs = np.hstack([rbf_hist_flat, local_obs])

        elif self.obs_mode == '2d_rbf_limited':
            in_range = (dm < self.comm_radius) & (0 < dm)
            nr_neighbors = np.sum(in_range)

            local_obs = self.get_local_obs()
            local_obs[-1] = nr_neighbors / (self.n_agents - 1)

            if np.any(in_range):
                dbn = np.stack([dm[in_range], my_orientation[in_range] + np.pi], axis=1)
                rbf_hist = U.get_weights_2d(dbn, self.mu, self.s,
                                            [self.bearing_bins, self.distance_bins]) / (self.n_agents - 1)

            else:
                rbf_hist = np.zeros([self.bearing_bins, self.distance_bins])

            rbf_hist_flat = rbf_hist.flatten()

            obs = np.hstack([rbf_hist_flat, local_obs])

        elif self.obs_mode == 'sum_obs_acc':
            in_range = (dm < self.comm_radius) & (0 < dm)
            nr_neighbors = np.sum(in_range)

            rel_vels = self.state.w_vel - vels

            local_obs = self.get_local_obs_acc()

            sum_obs = np.zeros(self.dim_rec_o)

            sum_obs[0:nr_neighbors, 0] = dm[in_range] / self.world_size
            sum_obs[0:nr_neighbors, 1] = np.cos(my_orientation[in_range])
            sum_obs[0:nr_neighbors, 2] = np.sin(my_orientation[in_range])
            sum_obs[:nr_neighbors, 3] = rel_vels[:, 0][in_range] / (2 * self.max_lin_velocity)
            sum_obs[:nr_neighbors, 4] = rel_vels[:, 1][in_range] / (2 * self.max_lin_velocity)
            sum_obs[0:nr_neighbors, 5] = 1
            sum_obs[0:self.n_agents - 1, 6] = 1

            obs = np.hstack([sum_obs.flatten(), local_obs])

        elif self.obs_mode == 'sum_obs_acc_full':
            in_range = (dm < self.comm_radius) & (0 < dm)
            nr_neighbors = np.sum(in_range)

            rel_vels = self.state.w_vel - vels

            local_obs = self.get_local_obs_acc()

            sum_obs = np.zeros(self.dim_rec_o)

            sum_obs[0:nr_neighbors, 0] = dm[in_range] / self.world_size
            sum_obs[0:nr_neighbors, 1] = np.cos(my_orientation[in_range])
            sum_obs[0:nr_neighbors, 2] = np.sin(my_orientation[in_range])
            sum_obs[0:nr_neighbors, 3] = np.cos(their_orientation[in_range])
            sum_obs[0:nr_neighbors, 4] = np.sin(their_orientation[in_range])
            sum_obs[:nr_neighbors, 5] = rel_vels[:, 0][in_range] / (2 * self.max_lin_velocity)
            sum_obs[:nr_neighbors, 6] = rel_vels[:, 1][in_range] / (2 * self.max_lin_velocity)
            sum_obs[0:nr_neighbors, 7] = 1
            sum_obs[0:self.n_agents - 1, 8] = 1

            obs = np.hstack([sum_obs.flatten(), local_obs])

        elif self.obs_mode == 'sum_obs_acc_no_vel':
            in_range = (dm < self.comm_radius) & (0 < dm)
            nr_neighbors = np.sum(in_range)

            local_obs = self.get_local_obs_acc()

            sum_obs = np.zeros(self.dim_rec_o)

            sum_obs[0:nr_neighbors, 0] = dm[in_range] / self.world_size
            sum_obs[0:nr_neighbors, 1] = np.cos(my_orientation[in_range])
            sum_obs[0:nr_neighbors, 2] = np.sin(my_orientation[in_range])
            sum_obs[0:nr_neighbors, 3] = 1
            sum_obs[0:self.n_agents - 1, 4] = 1

            obs = np.hstack([sum_obs.flatten(), local_obs])

        elif self.obs_mode == 'sum_obs_acc_limited':
            in_range = (dm < self.comm_radius) & (0 < dm)
            nr_neighbors = np.sum(in_range)

            rel_vels = self.state.w_vel - vels

            local_obs = self.get_local_obs_acc()
            local_obs[-1] = nr_neighbors / (self.n_agents - 1)

            sum_obs = np.zeros(self.dim_rec_o)

            sum_obs[0:nr_neighbors, 0] = dm[in_range] / self.world_size
            sum_obs[0:nr_neighbors, 1] = np.cos(my_orientation[in_range])
            sum_obs[0:nr_neighbors, 2] = np.sin(my_orientation[in_range])
            sum_obs[0:nr_neighbors, 3] = (nh_size[in_range] - nr_neighbors) / (self.n_agents - 2) if self.n_agents > 2\
                else np.zeros(nr_neighbors)
            sum_obs[:nr_neighbors, 4] = rel_vels[:, 0][in_range] / (2 * self.max_lin_velocity)
            sum_obs[:nr_neighbors, 5] = rel_vels[:, 1][in_range] / (2 * self.max_lin_velocity)
            sum_obs[0:nr_neighbors, 6] = 1
            sum_obs[0:self.n_agents - 1, 7] = 1

            obs = np.hstack([sum_obs.flatten(), local_obs])

        elif self.obs_mode == 'sum_obs':
            in_range = (dm < self.comm_radius) & (0 < dm)
            nr_neighbors = np.sum(in_range)

            local_obs = self.get_local_obs()
            local_obs[-1] = nr_neighbors / (self.n_agents - 1)

            sum_obs = np.zeros(self.dim_rec_o)

            sum_obs[0:nr_neighbors, 0] = dm[in_range] / self.world_size
            sum_obs[0:nr_neighbors, 1] = np.cos(my_orientation[in_range])
            sum_obs[0:nr_neighbors, 2] = np.sin(my_orientation[in_range])
            sum_obs[0:nr_neighbors, 3] = np.cos(their_orientation[in_range])
            sum_obs[0:nr_neighbors, 4] = np.sin(their_orientation[in_range])
            sum_obs[0:nr_neighbors, 5] = 1
            sum_obs[0:self.n_agents - 1, 6] = 1

            obs = np.hstack([sum_obs.flatten(), local_obs])

        elif self.obs_mode == 'sum_obs_limited':
            in_range = (dm < self.comm_radius) & (0 < dm)
            nr_neighbors = np.sum(in_range)

            local_obs = self.get_local_obs()
            local_obs[-1] = nr_neighbors / (self.n_agents - 1)

            sum_obs = np.zeros(self.dim_rec_o)

            sum_obs[0:nr_neighbors, 0] = dm[in_range] / self.world_size
            sum_obs[0:nr_neighbors, 1] = np.cos(my_orientation[in_range])
            sum_obs[0:nr_neighbors, 2] = np.sin(my_orientation[in_range])
            sum_obs[0:nr_neighbors, 3] = (nh_size[in_range] - nr_neighbors) / (self.n_agents - 2) if self.n_agents > 2\
                else np.zeros(nr_neighbors)
            sum_obs[0:nr_neighbors, 4] = np.cos(their_orientation[in_range])
            sum_obs[0:nr_neighbors, 5] = np.sin(their_orientation[in_range])
            sum_obs[0:nr_neighbors, 6] = 1
            sum_obs[0:self.n_agents - 1, 7] = 1

            obs = np.hstack([sum_obs.flatten(), local_obs])

        else:
            raise ValueError('histogram form must be 1D or 2D')

        return obs

    def get_local_obs_acc(self):
        local_obs = np.zeros(self.dim_local_o)
        local_obs[0] = self.state.p_vel[0] / self.max_lin_velocity
        local_obs[1] = self.state.p_vel[1] / self.max_ang_velocity

        if self.torus is False:
            if np.any(self.state.p_pos <= 1) or np.any(self.state.p_pos >= self.world_size - 1):
                wall_dists = np.array([self.world_size - self.state.p_pos[0],
                                       self.world_size - self.state.p_pos[1],
                                       self.state.p_pos[0],
                                       self.state.p_pos[1]])
                wall_angles = np.array([0, np.pi / 2, np.pi, 3 / 2 * np.pi]) - self.state.p_orientation
                closest_wall = np.argmin(wall_dists)
                local_obs[2] = wall_dists[closest_wall]
                local_obs[3] = np.cos(wall_angles[closest_wall])
                local_obs[4] = np.sin(wall_angles[closest_wall])
                # wall_angle2 = np.where(wall_angle > np.pi,
                #                        wall_angle - 2 * np.pi,
                #                        wall_angle)
                # local_obs[1] = 1
            else:
                local_obs[2] = 1
                local_obs[3:5] = 0

        return local_obs

    def get_local_obs(self):
        local_obs = np.zeros(self.dim_local_o)

        if self.torus is False:
            if np.any(self.state.p_pos <= 1) or np.any(self.state.p_pos >= self.world_size - 1):
                wall_dists = np.array([self.world_size - self.state.p_pos[0],
                                       self.world_size - self.state.p_pos[1],
                                       self.state.p_pos[0],
                                       self.state.p_pos[1]])
                wall_angles = np.array([0, np.pi / 2, np.pi, 3 / 2 * np.pi]) - self.state.p_orientation
                closest_wall = np.argmin(wall_dists)
                local_obs[0] = wall_dists[closest_wall]
                local_obs[1] = np.cos(wall_angles[closest_wall])
                local_obs[2] = np.sin(wall_angles[closest_wall])
            else:
                local_obs[0] = 1
                local_obs[1:3] = 0

        return local_obs
