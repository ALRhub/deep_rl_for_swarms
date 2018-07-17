import dill
import deep_rl_for_swarms.common.misc_util as mu
from deep_rl_for_swarms.policies import mlp_policy
from deep_rl_for_swarms.policies import mlp_policy_split
from deep_rl_for_swarms.policies import mlp_multi_policy_split
from deep_rl_for_swarms.policies import mlp_mean_embedding_policy
from deep_rl_for_swarms.policies import mlp_multi_mean_embedding_policy
from deep_rl_for_swarms.common.act_wrapper import ActWrapper
import numpy as np


with open('/tmp/env.pkl', 'rb') as file:
    env = dill.load(file)

if 'sum_obs' in env.obs_mode:
    if 'multi' not in env.obs_mode:
        def policy_fn(name, ob_space, ac_space):
            return mlp_mean_embedding_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                                       hid_size=[64], feat_size=[64])
    else:
        def policy_fn(name, ob_space, ac_space):
            return mlp_multi_mean_embedding_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                                             hid_size=[64], feat_size=[[64], [64]])

else:
    # if 'multi' not in env.obs_mode:
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy_split.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                          hid_size=[64, 64])
    # else:
    # def policy_fn(name, ob_space, ac_space):
    #     return mlp_multi_policy_split.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
    #                                             hid_size=[64],
    #                                             feat_size=[[64], [64]])

pi = ActWrapper.load("/tmp/model_390.pkl", policy_fn)

# Evaluate.
render_eval = True
render_mode = 'human'
nb_eval_steps = 1024
episodes = 1

for ep in range(episodes):
    ob = env.reset()
    if render_eval:
        env.render(mode=render_mode)

    for t_rollout in range(nb_eval_steps):
        ac, vpred = pi.act(False, np.stack(ob))
        ob, r, done, info = env.step(ac)
        if render_eval:
            env.render(mode=render_mode)
        if done or t_rollout == nb_eval_steps - 1:
            obs = env.reset()
            break
