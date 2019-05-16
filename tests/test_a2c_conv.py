import pytest
import tensorflow as tf
import numpy as np
from stable_baselines.a2c.utils import conv
from stable_baselines.common import tf_util
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.input import observation_input

ENV_ID = 'BreakoutNoFrameskip-v4'
SEED = 3
NUM_TIMESTEPS = 500
NUM_CPU = 2


@pytest.mark.parametrize("input1", [4])
@pytest.mark.parametrize("input2", [(3, 5)])
def test_conv_kernel(input1, input2):
    """
    test convolution kernel with various input formats

    :param input1: (int) for squared kernel matrix
    :param input2: (int, int) for non-squared kernel matrix
    """
    kwargs = {}
    n_envs = 1
    n_steps = 2
    n_batch = n_envs * n_steps
    scale = False
    env = VecFrameStack(make_atari_env(env_id=ENV_ID, num_env=n_envs, seed=SEED), 4)
    ob_space = env.observation_space

    graph = tf.Graph()
    with graph.as_default():
        sess = tf_util.make_session(graph=graph)
        _, scaled_images = observation_input(ob_space, n_batch, scale=scale)
        activ = tf.nn.relu
        layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=input1, stride=4
                             , init_scale=np.sqrt(2), **kwargs))
        layer_2 = activ(conv(layer_1, 'c2', n_filters=32, filter_size=input2, stride=4
                             , init_scale=np.sqrt(2), **kwargs))
        print(layer_2.shape)
    env.close()
