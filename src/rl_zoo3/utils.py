import argparse
import glob
import importlib
import os
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym as gym26
import gymnasium as gym
import stable_baselines3 as sb3  # noqa: F401
import torch as th  # noqa: F401
import yaml
from gymnasium import spaces
from huggingface_hub import HfApi
from huggingface_sb3 import EnvironmentName, ModelName
from sb3_contrib import ARS, QRDQN, TQC, TRPO, RecurrentPPO
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike  # noqa: F401
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack, VecNormalize

# For custom activation fn
from torch import nn as nn

# For custom feature extractor (adding self-attention layer)
import sys
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
import torch.nn.functional as F
from math import sqrt


ALGOS: Dict[str, Type[BaseAlgorithm]] = {
    "a2c": A2C,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    # SB3 Contrib,
    "ars": ARS,
    "qrdqn": QRDQN,
    "tqc": TQC,
    "trpo": TRPO,
    "ppo_lstm": RecurrentPPO,
}


class SelfAttnCNNPPO(BaseFeaturesExtractor):
    """
    Define our custom feature extractor (note: this is only the state representation block of the Actor-Critic network)
    For the CNN network here, we are following the PPO paper's setup.
    Note that the PPO paper follows the network architecture used in the both papers below.
    “Asynchronous methods for deep reinforcement learning”. In: arXiv preprint arXiv:1602.01783 (2016).
    "Playing atari with deep reinforcement learning". In NIPS Deep Learning Workshop. 2013.

    :param observation_space: (gym.Space) gym.spaces.Box
        This should be in the format of channel-first since the observation is an image for Atari envs.
    :param features_dim: (int) Number of features extracted (e.g. features from a CNN). default: 256 (follows the PPO paper)
        This corresponds to the number of unit for the last layer of the feature extractor.
        The last layer of feature extractor is also the first layer (input layer) of the net_arch block.
        We can also change the hidden layers of the net_arch by passing the net_arch parameter to policy_kwargs. 
        net_arch specifies a (fully-connected) network that maps the features to actions/value. Its architecture is controlled by the ``net_arch`` parameter.
        For more details, please visit https://github.com/DLR-RM/stable-baselines3/blob/378d197b00938579a6a5e04c739f41fec23fd805/docs/guide/custom_policy.rst
    :param self_attn: (str) Type of self-attention layer, default: 'NA' = No-Self-Attention. 
    :param ada_attn: (str) Type of adaptive attention configuration, default: None
    :param norm_softmax: (bool) Flag for whether normalization and softmax will be performed. default: True
        Here we follow the standard "Transformer"'s' self-attention definition. 
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int=256, self_attn: str='NA', ada_attn: str=None, norm_softmax: bool=True):
        super(SelfAttnCNNPPO, self).__init__(observation_space, features_dim)
        self.observation_space = observation_space
        self.n_input_channels = observation_space.shape[0] # this should be 4 if frame_stack is 4
        # self.features_dim = features_dim # this line causes "Can't set attribute" error
        self.self_attn = self_attn
        self.ada_attn = ada_attn
        self.norm_softmax = norm_softmax
        # self_attn type must not be None, pick one from ['NA', 'SWA', 'CWRA', 'CWCA', 'CWRCA']
        if self.self_attn is not None:
            if self.self_attn != 'NA':
                print("{} self-attention layer will be added.".format(self.self_attn))
                # initialize the given self-attention layer 
                self.self_attn_layer = MultiHeadAttention(size=16, self_attn=self.self_attn, ada_attn=self.ada_attn, norm_softmax=self.norm_softmax)
            else:
                print("No self-attention layer will be added.")
            # initialize all CNN layers
            self.c1 = nn.Conv2d(self.n_input_channels, 16, 8, stride=4) # default padding=0, output feature map size=(84-8)/4 + 1=20
            self.c2 = nn.Conv2d(16, 32, 4, stride=2) # output feature map size=(20-4)/2 + 1=9
            # get flattened layer size 
            self.n_flatten = self._get_flatten_size(self.observation_space.shape) # this should return 2592
            # initialize the first linear layer (i.e., the last layer of the feature extractor)
            self.l1 = nn.Linear(self.n_flatten, features_dim) # 2592*256
        else:
            sys.exit("self_attn argument must not be None! Please choose one from ['NA', 'SWA', 'CWRA', 'CWCA', 'CWRCA']")

    def custom_cnn_block(self, observations: th.Tensor) -> th.Tensor:
        h = F.relu(self.c1(observations)) # h's dimension: (batch, channel, height, width)
        # add self-attn layer between c1 and c2 based on the type of self-attention layer (only do this when self.self_attn is not 'NA') 
        if self.self_attn != 'NA':
            h = self.self_attn_layer(h) # pass h to 3 1x1 cnn kernels to generate q (query), k (key), v (value)
        h = F.relu(self.c2(h)) # h is of size (1,32,9,9) where n_flatten = 1*32*9*9=2592
        return h

    def _get_flatten_size(self, obs_shape: Tuple) -> int:
        with th.no_grad():
            # Compute flattened layer size by doing one forward pass through the custom_cnn_block 
            x = th.zeros(1, *obs_shape) # x is a zero tensor of size (1,4,84,84)
            h = self.custom_cnn_block(x) # h is of size (1,32,9,9) 
            return int(np.prod(h.size())) # np.prod([1,32,9,9]) = 2592

    def custom_feature_extractor(self, observations: th.Tensor) -> th.Tensor:
        h = self.custom_cnn_block(observations)
        h = h.reshape(-1).view(-1, self.n_flatten) # reshape(-1) converts h to a 1-D vector (row-major), view(-1,2592) converts a 1-D vector to a 2-D tensor (matrix) 
        h = F.relu(self.l1(h))
        return h

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.custom_feature_extractor(observations)


class MultiHeadAttention(nn.Module):
    """
    Define the multi-head attention block (we define 4 types of self-attention operations based on the feature dimensions selected for performing self-attention upon)
        'SWA': spatial-wise self-attention
        'CWRA': channel-wise-row self-attention
        'CWCA': channel-wise-column self-attention 
        'CWRCA': channel-wise-row-column self-attention
    """
    def __init__(self, size=16, self_attn=None, ada_attn=None, norm_softmax=True):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        assert self.self_attn is not None, "self_attn type cannot be None in multi-head attention block!"
        self.ada_attn = ada_attn
        self.norm_softmax = norm_softmax

        # initialize 3 1x1 CNN kernels for q,k,v generation 
        self.w_qs = nn.Conv2d(size, size, 1)
        self.w_ks = nn.Conv2d(size, size, 1)
        self.w_vs = nn.Conv2d(size, size, 1)

        # initialize self attn layer based on the type of self-attention operation
        if self.self_attn == 'SWA':
            self.self_attention = SpatialWiseAttention(require_scaling=self.norm_softmax)
        if self.self_attn == 'CWRA':
            self.self_attention = ChannelWiseRowAttention(require_scaling=self.norm_softmax)
        if self.self_attn == 'CWCA':
            self.self_attention = ChannelWiseColumnAttention(require_scaling=self.norm_softmax)
        if self.self_attn == 'CWRCA':
            self.self_attention = ChannelWiseRowColumnAttention(require_scaling=self.norm_softmax)

        # initialize adaptive attn coefficients regardless of user input (we will check user input in forward() function)
        self.alpha = th.nn.Parameter(th.zeros(1)) # initialize alpha = 0
        self.beta = th.nn.Parameter(th.ones(1)) # initialize beta = 1

    def forward(self, h: th.Tensor) -> th.Tensor:
        # print("multi-head attention forward starts")
        cnn_features = h # h = c1's output tensors with shape = (N,C,H,W)
        # get actual query, key, value tensors 
        q = self.w_qs(h)
        k = self.w_ks(h)
        v = self.w_vs(h) 

        attention = self.self_attention(q, k, v)

        # we can clamp the self.alpha and self.beta to be in the range of [0,1]
        # self.alpha = th.clamp(self.alpha, min=0.0, max=1.0)

        # based on user input on adaptive-attention setup, we define the output equation accordingly 
        if self.ada_attn is not None:
            if self.ada_attn == 'alpha':
                out = self.alpha*attention + cnn_features
            if self.ada_attn == 'beta':
                out = attention + self.beta*cnn_features
            if self.ada_attn == 'alpha_beta':
                out = self.alpha*attention + self.beta*cnn_features
        else:
            out = attention + cnn_features   
        # print("multi-head attention forward pass succeed")
        return out


class SpatialWiseAttention(nn.Module):
    """
    Perform self-attention on spatial dimensions (column and row), no permutation of q,k,v dimensions where q, k, v are of size (N, C, H, W) 
    """
    def __init__(self, require_scaling=True):
        super().__init__()
        self.require_scaling = require_scaling
        if self.require_scaling:
            self.SWA = ScaledDotProductAttention()
        else:
            self.SWA = DotProductAttention()

    def forward(self, q, k, v):
        if self.require_scaling:
            # get the size of the last dimension of non-permuted key 
            n = k.shape[-1] # width dim is the second dimension where k.shape=(32,32,20,20) 
            output = self.SWA(q, k, v, n)
        else:
            output = self.SWA(q, k, v)
        
        return output


class ChannelWiseRowAttention(nn.Module):
    """
    Perform self-attention on channel and row dimensions, permutation of q,k,v dimensions: (N, C, H, W) --> (N, W, H, C) 
    """
    def __init__(self, require_scaling=True):
        super().__init__()
        self.require_scaling = require_scaling
        if self.require_scaling:
            self.CWRA = ScaledDotProductAttention()
        else:
            self.CWRA = DotProductAttention()

    def forward(self, q, k, v):
        # permute dimensions so that the last 2 dims of q,k,v are (H,C)=(row, channel)
        q = q.permute(0, 3, 2, 1)
        k = k.permute(0, 3, 2, 1)
        v = v.permute(0, 3, 2, 1)

        if self.require_scaling:
            # get the channel length of permuted key 
            n = k.shape[-1] # channel dim is the last dimension 
            output = self.CWRA(q, k, v, n)
        else:
            output = self.CWRA(q, k, v)
        
        return output.permute(0, 3, 2, 1) # we need to convert (N, W, H, C) back to (N, C, H, W) before adding it to the cnn_features


class ChannelWiseColumnAttention(nn.Module):
    """
    Perform self-attention on channel and column dimensions, permutation of q,k,v dimensions: (N, C, H, W) --> (N, H, W, C) 
    """
    def __init__(self, require_scaling=True):
        super().__init__()
        self.require_scaling = require_scaling
        if self.require_scaling:
            self.CWCA = ScaledDotProductAttention()
        else:
            self.CWCA = DotProductAttention()

    def forward(self, q, k, v):
        # permute dimensions so that the last 2 dims of q,k,v are (W,C)=(column, channel)
        q = q.permute(0, 2, 3, 1)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 3, 1)

        if self.require_scaling:
            # get the channel length of permuted key 
            n = k.shape[-1] # channel dim is the last dimension 
            output = self.CWCA(q, k, v, n)
        else:
            output = self.CWCA(q, k, v)
        
        return output.permute(0, 3, 1, 2) # we need to convert (N, H, W, C) back to (N, C, H, W) before adding with cnn_features


class ChannelWiseRowColumnAttention(nn.Module):
    """
    Combine self-attention performed on both channel-row and channel-column dimensions 
    permutation of q,k,v dimensions: (N, C, H, W) --> (N, W, H, C) 
    permutation of q,k,v dimensions: (N, C, H, W) --> (N, H, W, C) 
    """
    def __init__(self, require_scaling=True):
        super().__init__()
        self.require_scaling = require_scaling
        if self.require_scaling:
            self.CWRA = ScaledDotProductAttention()
            self.CWCA = ScaledDotProductAttention()
        else:
            self.CWRA = DotProductAttention()
            self.CWCA = DotProductAttention()

    def forward(self, q, k, v):
        # permute dimensions so that the last 2 dims of q,k,v are (H,C)=(row, channel)
        q_r = q.permute(0, 3, 2, 1)
        k_r = k.permute(0, 3, 2, 1)
        v_r = v.permute(0, 3, 2, 1)
        # permute dimensions so that the last 2 dims of q,k,v are (W,C)=(column, channel)
        q_c = q.permute(0, 2, 3, 1)
        k_c = k.permute(0, 2, 3, 1)
        v_c = v.permute(0, 2, 3, 1)

        if self.require_scaling:
            # get the channel length of permuted key 
            n = k.shape[-1] # channel dim is the last dimension 
            output_r = self.CWRA(q_r, k_r, v_r, n).permute(0, 3, 2, 1) # convert back to (N, C, H, W)
            output_c = self.CWCA(q_c, k_c, v_c, n).permute(0, 3, 1, 2) # convert back to (N, C, H, W)
        else:
            output_r = self.CWRA(q_r, k_r, v_r).permute(0, 3, 2, 1) # convert back to (N, C, H, W)
            output_c = self.CWCA(q_c, k_c, v_c).permute(0, 3, 1, 2) # convert back to (N, C, H, W)
        
        return output_r + output_c # we can also make this adaptive 


class DotProductAttention(nn.Module):
    """
    Perform dot product attention (without normalization and softmax)
    """
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        attn = th.matmul(q, k.transpose(2, 3))
        output = th.matmul(attn, v)
        return output


class ScaledDotProductAttention(nn.Module):
    """
    Perform scaled dot product attention (with normalization and softmax)
    """
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, n):
        attn = th.matmul(q, k.transpose(2, 3))
        normed_attn = attn / sqrt(n)
        scaled_attn = F.softmax(normed_attn, dim=-1)
        output = th.matmul(scaled_attn, v)
        return output


def flatten_dict_observations(env: gym.Env) -> gym.Env:
    assert isinstance(env.observation_space, spaces.Dict)
    return gym.wrappers.FlattenObservation(env)


def get_wrapper_class(hyperparams: Dict[str, Any], key: str = "env_wrapper") -> Optional[Callable[[gym.Env], gym.Env]]:
    """
    Get one or more Gym environment wrapper class specified as a hyper parameter
    "env_wrapper".
    Works also for VecEnvWrapper with the key "vec_env_wrapper".

    e.g.
    env_wrapper: gym_minigrid.wrappers.FlatObsWrapper

    for multiple, specify a list:

    env_wrapper:
        - rl_zoo3.wrappers.PlotActionWrapper
        - rl_zoo3.wrappers.TimeFeatureWrapper


    :param hyperparams:
    :return: maybe a callable to wrap the environment
        with one or multiple gym.Wrapper
    """

    def get_module_name(wrapper_name):
        return ".".join(wrapper_name.split(".")[:-1])

    def get_class_name(wrapper_name):
        return wrapper_name.split(".")[-1]

    if key in hyperparams.keys():
        wrapper_name = hyperparams.get(key)

        if wrapper_name is None:
            return None

        if not isinstance(wrapper_name, list):
            wrapper_names = [wrapper_name]
        else:
            wrapper_names = wrapper_name

        wrapper_classes = []
        wrapper_kwargs = []
        # Handle multiple wrappers
        for wrapper_name in wrapper_names:
            # Handle keyword arguments
            if isinstance(wrapper_name, dict):
                assert len(wrapper_name) == 1, (
                    "You have an error in the formatting "
                    f"of your YAML file near {wrapper_name}. "
                    "You should check the indentation."
                )
                wrapper_dict = wrapper_name
                wrapper_name = list(wrapper_dict.keys())[0]
                kwargs = wrapper_dict[wrapper_name]
            else:
                kwargs = {}
            wrapper_module = importlib.import_module(get_module_name(wrapper_name))
            wrapper_class = getattr(wrapper_module, get_class_name(wrapper_name))
            wrapper_classes.append(wrapper_class)
            wrapper_kwargs.append(kwargs)

        def wrap_env(env: gym.Env) -> gym.Env:
            """
            :param env:
            :return:
            """
            for wrapper_class, kwargs in zip(wrapper_classes, wrapper_kwargs):
                env = wrapper_class(env, **kwargs)
            return env

        return wrap_env
    else:
        return None


def get_class_by_name(name: str) -> Type:
    """
    Imports and returns a class given the name, e.g. passing
    'stable_baselines3.common.callbacks.CheckpointCallback' returns the
    CheckpointCallback class.

    :param name:
    :return:
    """

    def get_module_name(name: str) -> str:
        return ".".join(name.split(".")[:-1])

    def get_class_name(name: str) -> str:
        return name.split(".")[-1]

    module = importlib.import_module(get_module_name(name))
    return getattr(module, get_class_name(name))


def get_callback_list(hyperparams: Dict[str, Any]) -> List[BaseCallback]:
    """
    Get one or more Callback class specified as a hyper-parameter
    "callback".
    e.g.
    callback: stable_baselines3.common.callbacks.CheckpointCallback

    for multiple, specify a list:

    callback:
        - rl_zoo3.callbacks.PlotActionWrapper
        - stable_baselines3.common.callbacks.CheckpointCallback

    :param hyperparams:
    :return:
    """

    callbacks: List[BaseCallback] = []

    if "callback" in hyperparams.keys():
        callback_name = hyperparams.get("callback")

        if callback_name is None:
            return callbacks

        if not isinstance(callback_name, list):
            callback_names = [callback_name]
        else:
            callback_names = callback_name

        # Handle multiple wrappers
        for callback_name in callback_names:
            # Handle keyword arguments
            if isinstance(callback_name, dict):
                assert len(callback_name) == 1, (
                    "You have an error in the formatting "
                    f"of your YAML file near {callback_name}. "
                    "You should check the indentation."
                )
                callback_dict = callback_name
                callback_name = list(callback_dict.keys())[0]
                kwargs = callback_dict[callback_name]
            else:
                kwargs = {}

            callback_class = get_class_by_name(callback_name)
            callbacks.append(callback_class(**kwargs))

    return callbacks


def create_test_env(
    env_id: str,
    n_envs: int = 1,
    stats_path: Optional[str] = None,
    seed: int = 0,
    log_dir: Optional[str] = None,
    should_render: bool = True,
    hyperparams: Optional[Dict[str, Any]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create environment for testing a trained agent

    :param env_id:
    :param n_envs: number of processes
    :param stats_path: path to folder containing saved running averaged
    :param seed: Seed for random number generator
    :param log_dir: Where to log rewards
    :param should_render: For Pybullet env, display the GUI
    :param hyperparams: Additional hyperparams (ex: n_stack)
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :return:
    """
    # Avoid circular import
    from rl_zoo3.exp_manager import ExperimentManager

    # Create the environment and wrap it if necessary
    assert hyperparams is not None
    env_wrapper = get_wrapper_class(hyperparams)

    hyperparams = {} if hyperparams is None else hyperparams

    if "env_wrapper" in hyperparams.keys():
        del hyperparams["env_wrapper"]

    vec_env_kwargs: Dict[str, Any] = {}
    vec_env_cls = DummyVecEnv
    if n_envs > 1 or (ExperimentManager.is_bullet(env_id) and should_render):
        # HACK: force SubprocVecEnv for Bullet env
        # as Pybullet envs does not follow gym.render() interface
        vec_env_cls = SubprocVecEnv  # type: ignore[assignment]
        # start_method = 'spawn' for thread safe

    # Fix for gym 0.26, to keep old behavior
    env_kwargs = env_kwargs or {}
    env_kwargs = deepcopy(env_kwargs)
    if "render_mode" not in env_kwargs and should_render:
        env_kwargs.update(render_mode="human")

    # Make Pybullet compatible with gym 0.26
    if ExperimentManager.is_bullet(env_id):
        spec = gym26.spec(env_id)
        env_kwargs.update(dict(apply_api_compatibility=True))
    else:
        # Define make_env here so it works with subprocesses
        # when the registry was modified with `--gym-packages`
        # See https://github.com/HumanCompatibleAI/imitation/pull/160
        try:
            spec = gym.spec(env_id)  # type: ignore[assignment]
        except gym.error.NameNotFound:
            # Registered with gym 0.26
            spec = gym26.spec(env_id)

    def make_env(**kwargs) -> gym.Env:
        env = spec.make(**kwargs)
        return env  # type: ignore[return-value]

    env = make_vec_env(
        make_env,
        n_envs=n_envs,
        monitor_dir=log_dir,
        seed=seed,
        wrapper_class=env_wrapper,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
    )

    if "vec_env_wrapper" in hyperparams.keys():
        vec_env_wrapper = get_wrapper_class(hyperparams, "vec_env_wrapper")
        assert vec_env_wrapper is not None
        env = vec_env_wrapper(env)  # type: ignore[assignment, arg-type]
        del hyperparams["vec_env_wrapper"]

    # Load saved stats for normalizing input and rewards
    # And optionally stack frames
    if stats_path is not None:
        if hyperparams["normalize"]:
            print("Loading running average")
            print(f"with params: {hyperparams['normalize_kwargs']}")
            path_ = os.path.join(stats_path, "vecnormalize.pkl")
            if os.path.exists(path_):
                env = VecNormalize.load(path_, env)
                # Deactivate training and reward normalization
                env.training = False
                env.norm_reward = False
            else:
                raise ValueError(f"VecNormalize stats {path_} not found")

        n_stack = hyperparams.get("frame_stack", 0)
        if n_stack > 0:
            print(f"Stacking {n_stack} frames")
            env = VecFrameStack(env, n_stack)
    return env


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    # Force conversion to float
    initial_value_ = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value_

    return func


def get_trained_models(log_folder: str) -> Dict[str, Tuple[str, str]]:
    """
    :param log_folder: Root log folder
    :return: Dict representing the trained agents
    """
    trained_models = {}
    for algo in os.listdir(log_folder):
        if not os.path.isdir(os.path.join(log_folder, algo)):
            continue
        for model_folder in os.listdir(os.path.join(log_folder, algo)):
            args_files = glob.glob(os.path.join(log_folder, algo, model_folder, "*/args.yml"))
            if len(args_files) != 1:
                continue  # we expect only one sub-folder with an args.yml file
            with open(args_files[0]) as fh:
                env_id = yaml.load(fh, Loader=yaml.UnsafeLoader)["env"]

            model_name = ModelName(algo, EnvironmentName(env_id))
            trained_models[model_name] = (algo, env_id)
    return trained_models


def get_hf_trained_models(organization: str = "sb3", check_filename: bool = False) -> Dict[str, Tuple[str, str]]:
    """
    Get pretrained models,
    available on the Hugginface hub for a given organization.

    :param organization: Huggingface organization
        Stable-Baselines (SB3) one is the default.
    :param check_filename: Perform additional check per model
        to be sure they match the RL Zoo convention.
        (this will slow down things as it requires one API call per model)
    :return: Dict representing the trained agents
    """
    api = HfApi()
    models = api.list_models(author=organization, cardData=True)

    trained_models = {}
    for model in models:
        # Try to extract algorithm and environment id from model card
        try:
            env_id = model.cardData["model-index"][0]["results"][0]["dataset"]["name"]
            algo = model.cardData["model-index"][0]["name"].lower()
            # RecurrentPPO alias is "ppo_lstm" in the rl zoo
            if algo == "recurrentppo":
                algo = "ppo_lstm"
        except (KeyError, IndexError):
            print(f"Skipping {model.modelId}")
            continue  # skip model if name env id or algo name could not be found

        env_name = EnvironmentName(env_id)
        model_name = ModelName(algo, env_name)

        # check if there is a model file in the repo
        if check_filename and not any(f.rfilename == model_name.filename for f in api.model_info(model.modelId).siblings):
            continue  # skip model if the repo contains no properly named model file

        trained_models[model_name] = (algo, env_id)

    return trained_models


def get_latest_run_id(log_path: str, env_name: EnvironmentName) -> int:
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :param log_path: path to log folder
    :param env_name:
    :return: latest run number
    """
    max_run_id = 0
    for path in glob.glob(os.path.join(log_path, env_name + "_[0-9]*")):
        run_id = path.split("_")[-1]
        path_without_run_id = path[: -len(run_id) - 1]
        if path_without_run_id.endswith(env_name) and run_id.isdigit() and int(run_id) > max_run_id:
            max_run_id = int(run_id)
    return max_run_id


def get_saved_hyperparams(
    stats_path: str,
    norm_reward: bool = False,
    test_mode: bool = False,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Retrieve saved hyperparameters given a path.
    Return empty dict and None if the path is not valid.

    :param stats_path:
    :param norm_reward:
    :param test_mode:
    :return:
    """
    hyperparams: Dict[str, Any] = {}
    if not os.path.isdir(stats_path):
        return hyperparams, None
    else:
        config_file = os.path.join(stats_path, "config.yml")
        if os.path.isfile(config_file):
            # Load saved hyperparameters
            with open(os.path.join(stats_path, "config.yml")) as f:
                hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            hyperparams["normalize"] = hyperparams.get("normalize", False)
        else:
            obs_rms_path = os.path.join(stats_path, "obs_rms.pkl")
            hyperparams["normalize"] = os.path.isfile(obs_rms_path)

        # Load normalization params
        if hyperparams["normalize"]:
            if isinstance(hyperparams["normalize"], str):
                normalize_kwargs = eval(hyperparams["normalize"])
                if test_mode:
                    normalize_kwargs["norm_reward"] = norm_reward
            else:
                normalize_kwargs = {"norm_obs": hyperparams["normalize"], "norm_reward": norm_reward}
            hyperparams["normalize_kwargs"] = normalize_kwargs
    return hyperparams, stats_path


class StoreDict(argparse.Action):
    """
    Custom argparse action for storing dict.

    In: args1:0.0 args2:"dict(a=1)"
    Out: {'args1': 0.0, arg2: dict(a=1)}
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        arg_dict = {}
        for arguments in values:
            key = arguments.split(":")[0]
            value = ":".join(arguments.split(":")[1:])
            # Evaluate the string as python code
            arg_dict[key] = eval(value)
        setattr(namespace, self.dest, arg_dict)


def get_model_path(
    exp_id: int,
    folder: str,
    algo: str,
    env_name: EnvironmentName,
    load_best: bool = False,
    load_checkpoint: Optional[str] = None,
    load_last_checkpoint: bool = False,
) -> Tuple[str, str, str]:
    if exp_id == 0:
        exp_id = get_latest_run_id(os.path.join(folder, algo), env_name)
        print(f"Loading latest experiment, id={exp_id}")
    # Sanity checks
    if exp_id > 0:
        log_path = os.path.join(folder, algo, f"{env_name}_{exp_id}")
    else:
        log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), f"The {log_path} folder was not found"

    model_name = ModelName(algo, env_name)

    if load_best:
        model_path = os.path.join(log_path, "best_model.zip")
        name_prefix = f"best-model-{model_name}"
    elif load_checkpoint is not None:
        model_path = os.path.join(log_path, f"rl_model_{load_checkpoint}_steps.zip")
        name_prefix = f"checkpoint-{load_checkpoint}-{model_name}"
    elif load_last_checkpoint:
        checkpoints = glob.glob(os.path.join(log_path, "rl_model_*_steps.zip"))
        if len(checkpoints) == 0:
            raise ValueError(f"No checkpoint found for {algo} on {env_name}, path: {log_path}")

        def step_count(checkpoint_path: str) -> int:
            # path follow the pattern "rl_model_*_steps.zip", we count from the back to ignore any other _ in the path
            return int(checkpoint_path.split("_")[-2])

        checkpoints = sorted(checkpoints, key=step_count)
        model_path = checkpoints[-1]
        name_prefix = f"checkpoint-{step_count(model_path)}-{model_name}"
    else:
        # Default: load latest model
        model_path = os.path.join(log_path, f"{env_name}.zip")
        name_prefix = f"final-model-{model_name}"

    found = os.path.isfile(model_path)
    if not found:
        raise ValueError(f"No model found for {algo} on {env_name}, path: {model_path}")

    return name_prefix, model_path, log_path
