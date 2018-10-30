import time
from deep_rl_for_swarms.common import dataset, explained_variance, fmt_row, zipsame
from deep_rl_for_swarms.common.act_wrapper import ActWrapper
from deep_rl_for_swarms.common.misc_util import flatten_lists
from deep_rl_for_swarms.common import tf_util as U
from deep_rl_for_swarms.common.mpi_adam import MpiAdam
from deep_rl_for_swarms.common.cg import cg
from deep_rl_for_swarms.common import colorize
from deep_rl_for_swarms.common.misc_util import copy_func
from deep_rl_for_swarms.common.mpi_moments import mpi_moments
from mpi4py import MPI
from common import logger
from contextlib import contextmanager
import numpy as np
import tensorflow as tf


def traj_segment_generator(pi, env, horizon, stochastic):
    # Initialize state variables
    t = 0
    n_agents = len(env.agents)
    ac = np.vstack([env.action_space.sample() for _ in range(n_agents)])
    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []
    time_steps = []

    # Initialize history arrays
    sub_sample_thresh = 8
    if n_agents > sub_sample_thresh:
        sub_sample = True
        sub_sample_idx = np.random.choice(n_agents, sub_sample_thresh, replace=False)

        obs = np.array([[ob[ssi] for ssi in sub_sample_idx] for _ in range(horizon)])
        rews = np.zeros([horizon, sub_sample_thresh], 'float32')
        vpreds = np.zeros([horizon, sub_sample_thresh], 'float32')
        news = np.zeros([horizon, sub_sample_thresh], 'int32')
        acs = np.array([ac[sub_sample_idx] for _ in range(horizon)])
        prevacs = acs.copy()
    else:
        sub_sample = False
        obs = np.array([ob for _ in range(horizon)])
        rews = np.zeros([horizon, n_agents], 'float32')
        vpreds = np.zeros([horizon, n_agents], 'float32')
        news = np.zeros([horizon, n_agents], 'int32')
        acs = np.array([ac for _ in range(horizon)])
        prevacs = acs.copy()

    while True:
        prevac = ac[sub_sample_idx] if sub_sample else ac
        ac, vpred = pi.act(stochastic, np.vstack(ob))
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            # yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
            #         "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
            #         "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            yield [
                dict(
                    ob=np.array(obs[:, na, :]),
                    rew=np.array(rews[:, na]),
                    vpred=np.array(vpreds[:, na]),
                    new=np.array(news[:, na]),
                    ac=np.array(acs[:, na, :]),
                    prevac=np.array(prevacs[:, na, :]),
                    nextvpred=vpred[na] * (1 - new) if not sub_sample else vpred[sub_sample_idx[na]] * (1 - new),
                    ep_rets=[epr[na] for epr in ep_rets],
                    ep_lens=ep_lens,
                    time_steps=np.array(time_steps)
                ) for na in range(min(n_agents, sub_sample_thresh))
            ]
            _, vpred = pi.act(stochastic, ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            time_steps = []
        i = t % horizon
        time_steps.append(t)
        obs[i] = [ob[ssi] for ssi in sub_sample_idx] if sub_sample else ob
        vpreds[i] = vpred[sub_sample_idx] if sub_sample else vpred
        news[i] = new
        acs[i] = ac[sub_sample_idx] if sub_sample else ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew[sub_sample_idx] if sub_sample else rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1


class TRPOAlgo():
    def __init__(self, env, policy_fn, *,
                 timesteps_per_batch, # what to train on
                 max_kl, cg_iters,
                 gamma, lam, # advantage estimation
                 entcoeff=0.0,
                 cg_damping=1e-2,
                 vf_stepsize=3e-4,
                 vf_iters =3,
                 max_timesteps=0, max_episodes=0, max_iters=0,  # time constraint
                 callback=None,
                 max_path_length=None
                 ):

        self.gamma = gamma
        self.gae_lambda = lam
        self.max_kl = max_kl
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.vf_stepsize = vf_stepsize
        self.vf_iters = vf_iters
        self.time_steps_per_batch = timesteps_per_batch
        if max_path_length is None:
            self.max_path_length = timesteps_per_batch
        else:
            self.max_path_length = max_path_length

        self.nworkers = MPI.COMM_WORLD.Get_size()
        self.rank = MPI.COMM_WORLD.Get_rank()
        np.set_printoptions(precision=3)
        # Setup losses and stuff
        # ----------------------------------------
        ob_space = env.observation_space
        ac_space = env.action_space
        pi = policy_fn("pi", ob_space, ac_space)
        oldpi = policy_fn("oldpi", ob_space, ac_space)
        atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
        ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return
        # n_size = tf.placeholder(dtype=tf.float32, shape=[None])  # neighborhood size

        ob = U.get_placeholder_cached(name="ob")
        ac = pi.pdtype.sample_placeholder([None])

        kloldnew = oldpi.pd.kl(pi.pd)
        ent = pi.pd.entropy()
        meankl = tf.reduce_mean(kloldnew)
        meanent = tf.reduce_mean(ent)
        entbonus = entcoeff * meanent

        vferr = tf.reduce_mean(tf.square(pi.vpred - ret))
        # pred_n_error = tf.reduce_mean(tf.square(pi.predict_n - n_size))

        ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # advantage * pnew / pold
        surrgain = tf.reduce_mean(ratio * atarg)

        optimgain = surrgain + entbonus  # - pred_n_error
        losses = [optimgain, meankl, entbonus, surrgain, meanent, vferr]  # , pred_n_error]
        self.loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy", "vf_loss"]  # , "pred_n_error"]

        dist = meankl

        all_var_list = pi.get_trainable_variables()
        var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
        var_list.extend([v for v in all_var_list if v.name.split("/")[1].startswith("me")])
        vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
        # vf_var_list.extend([v for v in all_var_list if v.name.split("/")[1].startswith("me")])
        self.vfadam = MpiAdam(vf_var_list)

        self.get_flat = U.GetFlat(var_list)
        self.set_from_flat = U.SetFromFlat(var_list)
        klgrads = tf.gradients(dist, var_list)
        flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
        shapes = [var.get_shape().as_list() for var in var_list]
        start = 0
        tangents = []
        for shape in shapes:
            sz = U.intprod(shape)
            tangents.append(tf.reshape(flat_tangent[start:start + sz], shape))
            start += sz
        gvp = tf.add_n(
            [tf.reduce_sum(g * tangent) for (g, tangent) in zipsame(klgrads, tangents)])  # pylint: disable=E1111
        fvp = U.flatgrad(gvp, var_list)

        self.assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                        for (oldv, newv) in
                                                        zipsame(oldpi.get_variables(), pi.get_variables())])
        self.compute_losses = U.function([ob, ac, atarg, ret], losses)
        self.compute_lossandgrad = U.function([ob, ac, atarg, ret], losses + [U.flatgrad(optimgain, var_list)])
        self.compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
        self.compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

        act_params = {
            'name': "pi",
            'ob_space': ob_space,
            'ac_space': ac_space,
        }

        self.pi = ActWrapper(pi, act_params)

        U.initialize()
        th_init = self.get_flat()
        MPI.COMM_WORLD.Bcast(th_init, root=0)
        self.set_from_flat(th_init)
        self.vfadam.sync()
        print("Init param sum", th_init.sum(), flush=True)

        # self.seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)
        if self.time_steps_per_batch > self.max_path_length:
            self.nr_traj_seg_gens = int(self.time_steps_per_batch / self.max_path_length)
            self.seg_gen = [copy_func(traj_segment_generator, "traj_seg_gen_{}".format(i))
                            (pi, env, timesteps_per_batch, stochastic=True) for i in range(self.nr_traj_seg_gens)]
        else:
            self.nr_traj_seg_gens = 1
            self.seg_gen = [traj_segment_generator(pi, env, self.time_steps_per_batch, stochastic=True)]

    def allmean(self, x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= self.nworkers
        return out

    @contextmanager
    def timed(self, msg):
        if self.rank == 0:
            logger.log(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            logger.log(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))
        else:
            yield

    def add_vtarg_and_adv(self, seg, gamma, lam):
        new = [np.append(p["new"], 0) for p in
               seg]  # last element is only used for last vtarg, but we already zeroed it if last new = 1
        vpred = [np.append(p["vpred"], p["nextvpred"]) for p in seg]

        for i, p in enumerate(seg):
            T = len(p["rew"])
            p["adv"] = gaelam = np.empty(T, 'float32')
            rew = p["rew"]
            lastgaelam = 0
            for t in reversed(range(T)):
                nonterminal = 1 - new[i][t + 1]
                delta = rew[t] + gamma * vpred[i][t + 1] * nonterminal - vpred[i][t]
                gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
            p["tdlamret"] = p["adv"] + p["vpred"]

    def train_once(self, iters_so_far):
        seg = []
        with self.timed("sampling"):
            for i in range(self.nr_traj_seg_gens):
                seg.extend(self.seg_gen[i].__next__())
        self.add_vtarg_and_adv(seg, self.gamma, self.gae_lambda)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        # ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        # vpredbefore = seg["vpred"] # predicted value function before udpate
        # atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        ob = np.concatenate([s['ob'] for s in seg], axis=0)
        ac = np.concatenate([s['ac'] for s in seg], axis=0)
        atarg = np.concatenate([s['adv'] for s in seg], axis=0)
        tdlamret = np.concatenate([s['tdlamret'] for s in seg], axis=0)
        vpredbefore = np.concatenate([s["vpred"] for s in seg], axis=0) # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

        # if hasattr(self.pi, "ret_rms"):
        #     self.pi.ret_rms.update(tdlamret)
        # if hasattr(self.pi, "ob_rms"):
        #     data_ob = ob[:, :-5].reshape([-1, 5])
        #     data_ob_extract = data_ob[np.where(data_ob[:, 3] == 1)][:, 0:-2]
        #     # data_ob = ob[:, :-1].reshape([-1, 7])[:, :-2]
        #     self.pi.ob_rms.update(data_ob_extract)  # update running mean/std for policy
        #     # self.pi.ob_rms.update(ob) # update running mean/std for policy

        # args = seg["ob"], seg["ac"], atarg
        args = ob, ac, atarg, tdlamret
        fvpargs = [arr[::5] for arr in args[:-1]]
        def fisher_vector_product(p):
            return self.allmean(self.compute_fvp(p, *fvpargs)) + self.cg_damping * p

        self.assign_old_eq_new() # set old parameter values to new parameter values
        with self.timed("computegrad"):
            *lossbefore, g = self.compute_lossandgrad(*args)
        lossbefore = self.allmean(np.array(lossbefore))
        g = self.allmean(g)
        if np.allclose(g, 0):
            logger.log("Got zero gradient. not updating")
        else:
            with self.timed("cg"):
                stepdir = cg(fisher_vector_product, g, cg_iters=self.cg_iters) # , verbose=self.rank==0)
            assert np.isfinite(stepdir).all()
            shs = .5*stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / self.max_kl)
            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = self.get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                self.set_from_flat(thnew)
                meanlosses = surr, kl, *_ = self.allmean(np.array(self.compute_losses(*args)))
                improve = surr - surrbefore
                logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                if not np.isfinite(meanlosses).all():
                    logger.log("Got non-finite value of losses -- bad!")
                elif kl > self.max_kl * 1.5:
                    logger.log("violated KL constraint. shrinking step.")
                elif improve < 0:
                    logger.log("surrogate didn't improve. shrinking step.")
                else:
                    logger.log("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                logger.log("couldn't compute a good step")
                self.set_from_flat(thbefore)
            if self.nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), self.vfadam.getflat().sum())) # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

        for (lossname, lossval) in zip(self.loss_names, meanlosses):
            logger.record_tabular(lossname, lossval)

        with self.timed("vf"):

            for _ in range(self.vf_iters):
                # for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                for (mbob, mbret) in dataset.iterbatches((ob, tdlamret),
                                                         include_final_partial_batch=False, batch_size=64):
                    g = self.allmean(self.compute_vflossandgrad(mbob, mbret))
                    self.vfadam.update(g, self.vf_stepsize)

        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        # lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        lrlocal = (seg[0]["ep_lens"], seg[0]["ep_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        average_action = np.mean(np.square(ac))
        logger.record_tabular('AverageReturn', np.mean(rews))
        logger.record_tabular('NumSamples', np.sum(lens) * len(seg))
        logger.record_tabular('StdReturn', np.std(rews))
        logger.record_tabular('MaxReturn', np.max(rews))
        logger.record_tabular('MinReturn', np.min(rews))
        logger.record_tabular('AverageAction', average_action)

        return rews, lens
