import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from deep_rl_for_swarms.ma_envs.commons.utils import EzPickle
from deep_rl_for_swarms.ma_envs import base
# from ma_envs.envs.environment import MultiAgentEnv
from deep_rl_for_swarms.ma_envs.agents.point_agents.rendezvous_agent import PointAgent
from deep_rl_for_swarms.ma_envs.commons import utils as U
import matplotlib.pyplot as plt


class RendezvousEnv(gym.Env, EzPickle):
    metadata = {'render.modes': ['human', 'animate']}

    def __init__(self, nr_agents=5,
                 obs_mode='sum_obs',
                 comm_radius=40,
                 world_size=100,
                 distance_bins=16,
                 bearing_bins=8,
                 torus=False,
                 dynamics='unicycle'):
        EzPickle.__init__(self, nr_agents, obs_mode, comm_radius, world_size, distance_bins, bearing_bins, torus, dynamics)
        self.nr_agents = nr_agents
        self.world_size = world_size
        self.obs_mode = obs_mode
        self.world = base.World(world_size, torus, dynamics)
        self.distance_bins = distance_bins
        self.bearing_bins = bearing_bins
        self.torus = torus
        self.dynamics = dynamics
        self.bounding_box = np.array([0., 2 * world_size, 0., 2 * world_size])
        self.comm_radius = comm_radius
        self.reward_mech = 'global'
        self.hist = None
        self.world.agents = [
            PointAgent(self) for _ in
            range(self.nr_agents)
        ]
        # self.seed()

        self.vel_hist = []
        self.state_hist = []
        self.timestep = 0
        self.ax = None

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
    def agents(self):
        return self.world.policy_agents

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    @property
    def timestep_limit(self):
        return 1024

    @property
    def is_terminal(self):
        # if (np.max(U.get_upper_triangle(self.world.distance_matrix,
        #                                 subtract_from_diagonal=-1)) < 1
        #     and np.mean([agent.state.p_vel**2 for agent in self.agents]) < 0.1**2)\
        #         or self.timestep >= self.timestep_limit:
        if self.timestep >= self.timestep_limit:
            # if self.ax:
            #     plt.close()
            return True
        else:
            return False

    def get_param_values(self):
        return self.__dict__

    def reset(self):
        self.timestep = 0
        # self.ax = None

        # self.nr_agents = np.random.randint(2, 10)
        # self.nr_agents = 10
        agent_states = np.random.rand(self.nr_agents, 3)
        agent_states[:, 0:2] = self.world_size * ((0.95 - 0.05) * agent_states[:, 0:2] + 0.05)
        agent_states[:, 2:3] = 2 * np.pi * agent_states[:, 2:3]

        self.world.agent_states = agent_states

        agent_list = [
            PointAgent(self)
            for _ in
            range(self.nr_agents)
        ]

        self.world.agents = agent_list
        self.world.reset()

        nr_agents_sensed = np.sum((0 < self.world.distance_matrix) &
                                  (self.world.distance_matrix < self.comm_radius), axis=1)  # / (self.nr_agents - 1)

        obs = []

        for i, bot in enumerate(agent_list):
            ob = bot.get_observation(self.world.distance_matrix[i, :],
                                     self.world.angle_matrix[i, :],
                                     self.world.angle_matrix[:, i],
                                     np.zeros([self.nr_agents, 2]),
                                     nr_agents_sensed
                                     )
            obs.append(ob)

        return obs

    def step(self, actions):

        self.timestep += 1

        # assert len(actions) == self.nr_agents
        # print(actions)
        clipped_actions = np.clip(actions[0:self.nr_agents, :], self.agents[0].action_space.low, self.agents[0].action_space.high)

        for agent, action in zip(self.agents, clipped_actions):
            agent.action.u = action

        self.world.step()

        next_obs = []

        velocities = np.vstack([agent.state.w_vel for agent in self.agents])
        # print(self.agents[0].state.p_vel)
        # self.vel_hist.append(velocities)
        nr_agents_sensed = np.sum((0 < self.world.distance_matrix) &
                                  (self.world.distance_matrix < self.comm_radius), axis=1)  # / (self.nr_agents - 1)

        for i, bot in enumerate(self.agents):
            ob = bot.get_observation(self.world.distance_matrix[i, :],
                                     self.world.angle_matrix[i, :],
                                     self.world.angle_matrix[:, i],
                                     velocities,
                                     nr_agents_sensed
                                     )
            next_obs.append(ob)

        rewards = self.get_reward(actions)

        done = self.is_terminal

        # if rewards[0] > -0.1 / self.comm_radius:  # distance of 0.1 in world coordinates, scaled by the reward scaling factor
        #     done = True

        # if done:
        #     rewards += 100
        # info = dict()
        info = {'state': self.world.agent_states, 'actions': actions, 'action_penalty': 0.05 * np.mean(actions**2),
                'velocities': np.vstack([agent.state.p_vel for agent in self.agents])}

        return next_obs, rewards, done, info

    def get_reward(self, actions):

        all_distances = U.get_upper_triangle(self.world.distance_matrix, subtract_from_diagonal=-1)
        all_distances_cap = np.where(all_distances > self.comm_radius, self.comm_radius, all_distances)
        all_distances_cap_norm = all_distances_cap / self.comm_radius  # (self.world_size * np.sqrt(2) / 2)
        dist_rew = np.mean(all_distances_cap_norm)
        action_pen = 0.001 * np.mean(actions**2)
        r = - dist_rew - action_pen
        r = np.ones((self.nr_agents,)) * r
        # print(dist_rew, action_pen)

        return r

    def render(self, mode='human'):  # , close=True):  check if works with older gym version
        if mode == 'animate':
            output_dir = "/tmp/video/"
            if self.timestep == 0:
                import shutil
                import os

                shutil.rmtree(output_dir)
                os.makedirs(output_dir, exist_ok=True)

        if not self.ax:
            fig, ax = plt.subplots()
            # ax.set_aspect('equal')
            # ax.set_xlim((0, self.world_size))
            # ax.set_ylim((0, self.world_size))
            self.ax = ax
            # self.fig2, self.axes = plt.subplots(1, 2)

        # else:
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_xlim((0, self.world_size))
        self.ax.set_ylim((0, self.world_size))
            # [ax.clear() for ax in self.axes]

        # self.ax.add_patch(
        #     patches.Rectangle(
        #         (5, 5),  # (x,y)
        #         self.world_size,  # width
        #         self.world_size,  # height
        #         fill=False
        #     )
        # )

        comm_circles = []
        self.ax.scatter(self.world.agent_states[:, 0], self.world.agent_states[:, 1], c='b', s=10)
        # self.ax.scatter(self.nodes_all[:, 0], self.nodes_all[:, 1], c='k')
        # self.ax.scatter(self.center_of_mass[0], self.center_of_mass[1], c='g')
        # self.ax.scatter(self.center_of_mass_torus[0], self.center_of_mass_torus[1], c='r')
        # self.ax.plot(self.actors[:, 0], self.actors[:, 1], marker=(3, 0, self.actors[:, 2]), markersize=20, linestyle='None')
        for i in range(self.nr_agents):
            # self.ax.plot(self.actors[i, 0], self.actors[i, 1], marker=(3, 0, self.actors[i, 2]/np.pi*180-90), markersize=20,
            #              linestyle='None', color='g' if i != 0 else 'b')
            comm_circles.append(plt.Circle((self.world.agent_states[i, 0],
                                            self.world.agent_states[i, 1]),
                                           self.comm_radius, color='g' if i != 0 else 'b', fill=False))

            self.ax.add_artist(comm_circles[i])

            # self.ax.text(self.world.agent_states[i, 0], self.world.agent_states[i, 1],
            #              i, ha='center',
            #              va='center', size=25)
        # circles.append(plt.Circle((self.evader[0],
        #                            self.evader[1]),
        #                           self.evader_radius, color='r', fill=False))
        # self.ax.add_artist(circles[-1])
        # self.axes[0].imshow(self.agents[0].histogram[0, :, :], vmin=0, vmax=10)
        # self.axes[1].imshow(self.agents[0].histogram[1, :, :], vmin=0, vmax=1)
        if mode == 'human':
            plt.pause(0.01)
        elif mode == 'animate':
            if self.timestep % 2 == 0:
                plt.savefig(output_dir + format(self.timestep//2, '04d'))

            if self.is_terminal:
                import os
                os.system("ffmpeg -r 10 -i " + output_dir + "%04d.png -c:v libx264 -pix_fmt yuv420p -y /tmp/out.mp4")


if __name__ == '__main__':
    n_ag = 10
    env = RendezvousEnv(nr_agents=n_ag,
                        obs_mode='3d_rbf',
                        comm_radius=40,
                        world_size=100,
                        distance_bins=8,
                        bearing_bins=8,
                        dynamics='unicycle',
                        torus=False)
    for e in range(100):
        o = env.reset()
        dd = False
        flip = -1
        for t in range(10):
            a = 2 * np.random.rand(n_ag, 2) - 1
            # print(t, flip, env.agents[0].state.p_vel)
            # if t % 50 == 0:
            #     flip = -flip
            # a[:, 0] = 1 * flip
            # a[:, 1] = 0
            # if t >= 150:
            #     a = np.zeros([20, 2])
            #     print(t, flip, env.agents[0].state.p_vel)
            a[:, 0] = 1
            # a[:, 1] = 1
            # if t >= 60:
            #     a = np.zeros([20, 2])
            o, rew, dd, _ = env.step(a)
            # if rew.sum() < 0:
            #     print(rew[0])
            if t % 1 == 0:
                env.render(mode='human')
                # time.sleep(0.5)
            if dd:

                break
        print(np.mean(env.agents[0].neighborhood_size_hist))
