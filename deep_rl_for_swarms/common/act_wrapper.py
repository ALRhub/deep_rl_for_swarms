import zipfile
import cloudpickle
import tempfile
import os
from common import logger
import numpy as np
import tensorflow as tf


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params
        self.saver = tf.train.Saver()

    @staticmethod
    def load(path, pol_fn):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        if "dim_rec_o" in act_params:
            act_params['ob_space'].dim_rec_o = act_params["dim_rec_o"]
            act_params['ob_space'].dim_local_o = act_params['ob_space'].shape[0] - np.prod(act_params["dim_rec_o"])
            del act_params["dim_rec_o"]

        act = pol_fn(**act_params)
        sess = tf.Session()
        sess.__enter__()
        aw = ActWrapper(act, act_params)
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            aw.load_state(os.path.join(td, "model"))

        return aw

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def act(self, *args, **kwargs):
        return self._act.act(*args, **kwargs)

    def load_state(self, fname):
        self.saver.restore(tf.get_default_session(), fname)

    def save_state(self, fname):
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        self.saver.save(tf.get_default_session(), fname)

    @property
    def recurrent(self):
        return self._act.recurrent

    @property
    def ob_rms(self):
        if hasattr(self._act, "ob_rms"):
            return self._act.ob_rms
        else:
            return None

    @property
    def ret_rms(self):
        if hasattr(self._act, "ret_rms"):
            return self._act.ret_rms
        else:
            return None

    def save(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            self.save_state(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)