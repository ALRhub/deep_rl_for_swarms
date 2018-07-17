import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from deep_rl_for_swarms.ma_envs.commons.utils import EzPickle
from deep_rl_for_swarms.ma_envs import base
# from ma_envs.envs.environment import MultiAgentEnv
from deep_rl_for_swarms.ma_envs.agents.point_agents.pursuer_agent import PointAgent
from deep_rl_for_swarms.ma_envs.agents.point_agents.evader_agent import Evader
from deep_rl_for_swarms.ma_envs.commons import utils as U
import networkx as nwx
import itertools
try:
    import matplotlib
    # matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as mpla
    from matplotlib.patches import Wedge
    from matplotlib.patches import RegularPolygon
    import matplotlib.patches as patches
except:
    pass


class PursuitEvasionEnv(gym.Env, EzPickle):
    metadata = {'render.modes': ['human', 'animate']}

    def __init__(self,
                 nr_pursuers=5,
                 nr_evaders=1,
                 obs_mode='2D_rbf',
                 comm_radius=40,
                 world_size=100,
                 distance_bins=8,
                 bearing_bins=8,
                 torus=True,
                 dynamics='direct'):
        EzPickle.__init__(self, nr_pursuers, nr_evaders, obs_mode, comm_radius, world_size, distance_bins,
                          bearing_bins, torus, dynamics)
        self.nr_agents = nr_pursuers
        self.nr_evaders = 1
        self.obs_mode = obs_mode
        self.distance_bins = distance_bins
        self.bearing_bins = bearing_bins
        self.comm_radius = comm_radius
        self.obs_radius = comm_radius / 2
        self.torus = torus
        self.dynamics = dynamics
        self.world_size = world_size
        self.world = base.World(world_size, torus, dynamics)
        self.world.agents = [
            PointAgent(self) for _ in
            range(self.nr_agents)
        ]
        [self.world.agents.append(Evader(self)) for _ in range(self.nr_evaders)]
        self._reward_mech = 'global'
        self.timestep = None
        self.hist = None
        self.ax = None
        self.obs_comm_matrix = None
        if self.obs_mode == 'sum_obs_learn_comm':
            self.world.dim_c = 1
        # self.seed()

    @property
    def state_space(self):
        return spaces.Box(low=-10., high=10., shape=(self.nr_agents * 3,), dtype=np.float32)

    @property
    def observation_space(self):
        return self.agents[0].observation_space

    @property
    def action_space(self):
        return self.agents[0].action_space

    @property
    def reward_mech(self):
        return self.reward_mech

    @property
    def agents(self):
        return self.world.policy_agents

    def get_param_values(self):
        return self.__dict__

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    @property
    def timestep_limit(self):
        return 1024

    @property
    def is_terminal(self):
        if self.timestep >= self.timestep_limit:
            if self.ax:
                plt.close()
            return True
        return False

    def reset(self):
        self.timestep = 0
        # self.ax = None

        # self.nr_agents = 5  # np.random.randint(2, 10)
        self.world.agents = [
            PointAgent(self)
            for _ in
            range(self.nr_agents)
        ]

        self.world.agents.append(Evader(self))

        self.obs_comm_matrix = self.obs_radius * np.ones([self.nr_agents + 1, self.nr_agents + 1])
        self.obs_comm_matrix[0:-self.nr_evaders, 0:-self.nr_evaders] = self.comm_radius

        pursuers = np.random.rand(self.nr_agents, 3)
        pursuers[:, 0:2] = self.world_size * ((0.95 - 0.05) * pursuers[:, 0:2] + 0.05)
        pursuers[:, 2:3] = 2 * np.pi * pursuers[:, 2:3]

        evader = (0.95 - 0.05) * np.random.rand(self.nr_evaders, 2) + 0.05
        evader = self.world_size * evader

        self.world.agent_states = pursuers
        self.world.landmark_states = evader
        self.world.reset()

        if self.obs_radius < self.world_size * np.sqrt(2):
            sets = self.graph_feature()

        feats = [p.graph_feature for p in self.agents]

        if self.world.dim_c > 0:
            messages = np.zeros([self.nr_agents, 1])
        else:
            messages = []

        obs = []

        for i, bot in enumerate(self.world.policy_agents):
            # bot_in_subset = [list(s) for s in sets if i in s]
            # [bis.remove(i) for bis in bot_in_subset]
            ob = bot.get_observation(self.world.distance_matrix[i, :],
                                     self.world.angle_matrix[i, :],
                                     self.world.angle_matrix[:, i],
                                     feats,
                                     np.zeros([self.nr_agents, 2])
                                     )
            obs.append(ob)

        return obs

    def step(self, actions):

        self.timestep += 1

        assert len(actions) == self.nr_agents
        # print(actions)
        clipped_actions = np.clip(actions, self.agents[0].action_space.low, self.agents[0].action_space.high)

        for agent, action in zip(self.agents, clipped_actions):
            agent.action.u = action[0:2]
            if self.world.dim_c > 0:
                agent.action.c = action[2:]

        self.world.step()

        if self.obs_radius < self.world_size * np.sqrt(2):
            sets = self.graph_feature()

        feats = [p.graph_feature for p in self.agents]

        if self.world.dim_c > 0:
            messages = clipped_actions[:, 2:]
        else:
            messages = []

        velocities = np.vstack([agent.state.w_vel for agent in self.agents])

        next_obs = []

        for i, bot in enumerate(self.world.policy_agents):
            # print(hop_counts)
            # bot_in_subset = [list(s) for s in sets if i in s]
            # [bis.remove(i) for bis in bot_in_subset]
            ob = bot.get_observation(self.world.distance_matrix[i, :],
                                     self.world.angle_matrix[i, :],
                                     self.world.angle_matrix[:, i],
                                     feats,
                                     velocities
                                     )
            next_obs.append(ob)

        rewards = self.get_reward(actions)

        done = self.is_terminal
        if rewards[0] > -1 / self.obs_radius:  # distance of 1 in world coordinates, scaled by the reward scaling factor
            done = True
        # if done and self.timestep < self.timestep_limit:
        #     rewards = 100 * np.ones((self.nr_agents,))
        # info = dict()
        info = {'pursuer_states': self.world.agent_states,
                'evader_states': self.world.landmark_states,
                'state': np.vstack([self.world.agent_states[:, 0:2], self.world.landmark_states]),
                'actions': actions}

        return next_obs, rewards, done, info

    def get_reward(self, actions):
        r = -np.minimum(np.min(self.world.distance_matrix[-1, :-self.nr_evaders]), self.obs_radius) / self.obs_radius  # - 0.05 * np.sum(np.mean(actions**2, axis=1))
        # r = -np.minimum(np.partition(self.world.distance_matrix[-1, :-self.nr_evaders], 2)[2], self.obs_radius) / self.world_size
        # r = - 1
        # print(np.min(self.world.distance_matrix[-1, :-self.nr_evaders]))
        r = np.ones((self.nr_agents,)) * r

        return r

    def graph_feature(self):
        adj_matrix = np.array(self.world.distance_matrix < self.obs_comm_matrix, dtype=float)
        # visibles = np.sum(adj_matrix, axis=0) - 1
        # print("mean neighbors seen: ", np.mean(visibles[:-1]))
        # print("evader seen by: ", visibles[-1])
        sets = U.dfs(adj_matrix, 2)

        g = nwx.Graph()

        for set_ in sets:
            l_ = list(set_)
            if self.nr_agents in set_:
                # points = self.nodes[set_, 0:2]
                # dist_matrix = self.get_euclid_distances(points, matrix=True)

                # determine distance and adjacency matrix of subset
                dist_matrix = np.array([self.world.distance_matrix[x] for x in list(itertools.product(l_, l_))]).reshape(
                        [len(l_), len(l_)])

                obs_comm_matrix = np.array(
                    [self.obs_comm_matrix[x] for x in list(itertools.product(l_, l_))]).reshape(
                    [len(l_), len(l_)])

                adj_matrix_sub = np.array((0 <= dist_matrix) & (dist_matrix < obs_comm_matrix), dtype=float)
                connection = np.where(adj_matrix_sub == 1)
                edges = [[x[0], x[1]] for x in zip([l_[c] for c in connection[0]], [l_[c] for c in connection[1]])]

                g.add_nodes_from(l_)
                g.add_edges_from(edges)
                for ind, e in enumerate(edges):
                    g[e[0]][e[1]]['weight'] = dist_matrix[connection[0][ind], connection[1][ind]]

        for i in range(self.nr_agents):
            try:
                self.agents[i].graph_feature = \
                    nwx.shortest_path_length(g, source=i, target=self.nr_agents, weight='weight')
            except:
                self.agents[i].graph_feature = np.inf

        return sets

    def render(self, mode='human'):
        if mode == 'animate':
            output_dir = "/tmp/video/"
            if self.timestep == 0:
                import shutil
                import os
                try:
                    shutil.rmtree(output_dir)
                except FileNotFoundError:
                    pass
                os.makedirs(output_dir, exist_ok=True)

        if not self.ax:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.set_xlim((0, self.world_size))
            ax.set_ylim((0, self.world_size))
            self.ax = ax

        else:
            self.ax.clear()
            self.ax.set_aspect('equal')
            self.ax.set_xlim((0, self.world_size))
            self.ax.set_ylim((0, self.world_size))

        comm_circles = []
        obs_circles = []
        self.ax.scatter(self.world.landmark_states[:, 0], self.world.landmark_states[:, 1], c='r', s=20)
        self.ax.scatter(self.world.agent_states[:, 0], self.world.agent_states[:, 1], c='b', s=20)
        for i in range(self.nr_agents):
            comm_circles.append(plt.Circle((self.world.agent_states[i, 0],
                                       self.world.agent_states[i, 1]),
                                      self.comm_radius, color='g', fill=False))
            self.ax.add_artist(comm_circles[i])

            obs_circles.append(plt.Circle((self.world.agent_states[i, 0],
                                            self.world.agent_states[i, 1]),
                                           self.obs_radius, color='g', fill=False))
            self.ax.add_artist(obs_circles[i])

            # self.ax.text(self.world.agent_states[i, 0], self.world.agent_states[i, 1],
            #              "{}".format(i), ha='center',
            #              va='center', size=20)
        # circles.append(plt.Circle((self.evader[0],
        #                            self.evader[1]),
        #                           self.evader_radius, color='r', fill=False))
        # self.ax.add_artist(circles[-1])

        if mode == 'human':
            plt.pause(0.01)
        elif mode == 'animate':
            if self.timestep % 1 == 0:
                plt.savefig(output_dir + format(self.timestep//1, '04d'))

            if self.is_terminal:
                import os
                os.system("ffmpeg -r 10 -i " + output_dir + "%04d.png -c:v libx264 -pix_fmt yuv420p -y /tmp/out.mp4")


if __name__ == '__main__':
    nr_pur = 10
    env = PursuitEvasionEnv(nr_pursuers=nr_pur,
                            nr_evaders=1,
                            obs_mode='sum_obs_no_ori',
                            comm_radius=200 * np.sqrt(2),
                            world_size=100,
                            distance_bins=8,
                            bearing_bins=8,
                            dynamics='unicycle',
                            torus=True)
    for ep in range(1):
        o = env.reset()
        dd = False
        for t in range(1024):
            a = 1 * np.random.randn(nr_pur, env.world.agents[0].dim_a)
            a[:, 0] = 1
            # a[:, 1] = 0
            o, rew, dd, _ = env.step(a)
            # if rew.sum() < 0:
            #     print(rew[0])
            if t % 1 == 0:
                env.render()

            if dd:
                break
