"""
Helpers for scripts like run_atari.py.
"""

import os
import gym
import deep_rl_for_swarms.ma_envs
from deep_rl_for_swarms.common import logger
from deep_rl_for_swarms.common.bench.monitor import Monitor
from deep_rl_for_swarms.common.misc_util import set_global_seeds


def make_multiagent_env(env_id, seed, rank=0):
    set_global_seeds(seed)
    env = gym.make(env_id)
    env = Monitor(env, logger.get_dir())
    env.seed(seed)
    return env

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

