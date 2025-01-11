# The implementation is based on:
# https://github.com/sp-uhh/sgmse
# Licensed under MIT


import math

import torch

from abc import ABC, abstractmethod
from typing import Tuple
import einops

import espnet2.enh.diffusion.sampling as sampling
from espnet2.enh.diffusion.abs_diffusion import AbsDiffusion
from espnet2.enh.diffusion.sdes import OUVESDE, OUVPSDE, SDE
from espnet2.enh.layers.dcunet import DCUNet
from espnet2.enh.layers.ncsnpp import NCSNpp
from espnet2.train.class_choices import ClassChoices
from espnet2.enh.separator.se_mamba_separator import SEMambaSeparator

score_choices = ClassChoices(
    name="score_model",
    classes=dict(dcunet=DCUNet, ncsnpp=NCSNpp),
    type_check=torch.nn.Module,
    default=None,
)

sde_choices = ClassChoices(
    name="sde",
    classes=dict(
        ouve=OUVESDE,
        ouvp=OUVPSDE,
    ),
    type_check=SDE,
    default="ouve",
)


class ScoreModel(AbsDiffusion):
    def __init__(self, **kwargs):
        super().__init__()

        score_model = kwargs["score_model"]  # noqa
        score_model_class = score_choices.get_class(kwargs["score_model"])
        self.dnn = score_model_class(**kwargs["score_model_conf"])
        self.sde = sde_choices.get_class(kwargs["sde"])(**kwargs["sde_conf"])
        self.loss_type = getattr(kwargs, "loss_type", "mse")
        self.t_eps = getattr(kwargs, "t_eps", 3e-2)

    def _loss(self, err):
        if self.loss_type == "mse":
            losses = torch.square(err.abs())
        elif self.loss_type == "mae":
            losses = err.abs()
        # taken from reduce_op function: sum over channels and position
        # and mean over batch dim presumably only important for absolute
        # loss number, not for gradients
        loss = torch.mean(0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss

    def get_pc_sampler(
        self, predictor_name, corrector_name, y, N=None, minibatch=None, **kwargs
    ):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(
                predictor_name,
                corrector_name,
                sde=sde,
                score_fn=self.score_fn,
                y=y,
                **kwargs
            )
        else:
            M = y.shape[0]

            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(math.ceil(M / minibatch))):
                    y_mini = y[i * minibatch : (i + 1) * minibatch]
                    sampler = sampling.get_pc_sampler(
                        predictor_name,
                        corrector_name,
                        sde=sde,
                        score_fn=self.score_fn,
                        y=y_mini,
                        **kwargs
                    )
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns

            return batched_sampling_fn

    def get_ode_sampler(self, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(
                sde, self.score_fn, y=y, device=y.device, **kwargs
            )
        else:
            M = y.shape[0]

            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(math.ceil(M / minibatch))):
                    y_mini = y[i * minibatch : (i + 1) * minibatch]
                    sampler = sampling.get_ode_sampler(
                        sde, self.score_fn, y=y_mini, device=y_mini.device, **kwargs
                    )
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return sample, ns

            return batched_sampling_fn

    def score_fn(self, x, t, y):
        # Concatenate y as an extra channel
        dnn_input = torch.cat([x, y], dim=1)

        # the minus is most likely unimportant here - taken from Song's repo
        score = -self.dnn(dnn_input, t)
        return score

    def forward(
        self,
        feature_ref,
        feature_mix,
    ):
        # feature_ref: B, T, F
        # feature_mix: B, T, F
        x = feature_ref.permute(0, 2, 1).unsqueeze(1)
        y = feature_mix.permute(0, 2, 1).unsqueeze(1)

        t = (
            torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps)
            + self.t_eps
        )
        mean, std = self.sde.marginal_prob(x, t, y)
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
        sigmas = std[:, None, None, None]
        perturbed_data = mean + sigmas * z

        score = self.score_fn(perturbed_data, t, y)
        assert score.shape == x.shape, "Check the output shape of the score_fn."
        err = score * sigmas + z
        loss = self._loss(err)

        return loss

    def enhance(
        self,
        noisy_specturm,
        sampler_type="pc",
        predictor="reverse_diffusion",
        corrector="ald",
        N=30,
        corrector_steps=1,
        snr=0.5,
        **kwargs
    ):
        """Enhance function.

        Args:
            noisy_specturm (torch.Tensor): noisy feature in [Batch, T, F]
            sampler_type (str): sampler, 'pc' for Predictor-Corrector and 'ode' for ODE
                                sampler.
            predictor (str): the name of Predictor. 'reverse_diffusion',
                            'euler_maruyama', or 'none'
            corrector (str): the name of Corrector. 'langevin', 'ald' or 'none'
            N (int): The number of reverse sampling steps.
            corrector_steps (int) : number of steps in the Corrector.
            snr (float): The SNR to use for the corrector.
        Returns:
            X_Hat (torch.Tensor): enhanced feature in [Batch, T, F]
        """
        Y = noisy_specturm.permute(0, 2, 1).unsqueeze(1)
        if sampler_type == "pc":
            sampler = self.get_pc_sampler(
                predictor,
                corrector,
                Y,
                N=N,
                corrector_steps=corrector_steps,
                snr=snr,
                intermediate=False,
                **kwargs
            )
        elif sampler_type == "ode":
            sampler = self.get_ode_sampler(Y, N=N, **kwargs)
        else:
            print("{} is not a valid sampler type!".format(sampler_type))

        X_Hat, nfe = sampler()

        X_Hat = X_Hat.squeeze(1).permute(0, 2, 1)

        return X_Hat


class FlowModel(AbsDiffusion):
    def __init__(self, **kwargs):
        super().__init__()

        estimator_model = kwargs["estimator_model"]  # noqa
        estimator_model_class = flow_estimator_choices.get_class(kwargs["estimator_model"])
        self.estimator = estimator_model_class(**kwargs["estimator_model_conf"])
        self.flow = flow_choices.get_class(kwargs["flow"])(**kwargs["flow_conf"])
        self.loss_type = getattr(kwargs, "loss_type", "mse")

    def _loss(self, err):
        if self.loss_type == "mse":
            losses = torch.square(err.abs())
        elif self.loss_type == "mae":
            losses = err.abs()
        # taken from reduce_op function: sum over channels and position
        # and mean over batch dim presumably only important for absolute
        # loss number, not for gradients
        loss = torch.mean(0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss

    def forward(
        self,
        feature_ref,
        feature_mix,
    ):
        # feature_ref: B, T, F
        # feature_mix: B, T, F
        x = feature_ref.permute(0, 2, 1).unsqueeze(1)
        y = feature_mix.permute(0, 2, 1).unsqueeze(1)


        # flow starts from zero
        x_start = torch.zeros_like(y)

        # generate time
        t = self.flow.generate_time(batch_size=x.shape[0]).to(device=x.device)
        sample = self.flow.sample(time=t, x_start=x_start, x_end=x)

        # estimator input: concatenate sample and y (noisy spectrogram)
        estimator_input = torch.cat([sample, y], dim=1)

        # estimate the vector field using the neural estimator
        # note: ilense are not provided to enhance
        estimate, *_ = self.estimator(input=estimator_input, ilens=None, time_cond=t)
        # take the first element
        assert len(estimate) == 1, "Only single-channel mixture is supported now"
        estimate = estimate[0].unsqueeze(1)

        # actual vector field
        conditional_vector_field = self.flow.vector_field(time=t, x_start=x_start, x_end=x, point=sample)

        # estimation error
        err = estimate - conditional_vector_field

        loss = self._loss(err)

        return loss
    

    def get_sampler(self, num_steps=None, minibatch=None, **kwargs):
        """
        Get a sampler for the flow model.
        """
        if minibatch is None:
            return ConditionalFlowMatchingEulerSampler(estimator=self.estimator, num_steps=num_steps, **kwargs)
        else:
            raise NotImplementedError("Minibatch sampling not implemented for flow model")

    def enhance(
        self,
        noisy_specturm,
        N=5,
        **kwargs
    ):
        """Enhance function.

        Args:
            noisy_specturm (torch.Tensor): noisy feature in [Batch, T, F]
            N (int): The number of reverse sampling steps.
            corrector_steps (int) : number of steps in the Corrector.
            snr (float): The SNR to use for the corrector.
        Returns:
            X_Hat (torch.Tensor): enhanced feature in [Batch, T, F]
        """
        Y = noisy_specturm.permute(0, 2, 1).unsqueeze(1)
        sampler = self.get_sampler(num_steps=N, **kwargs)

        init_state = torch.randn_like(Y) * self.flow.sigma_start

        # Sampler
        X_Hat, _ = sampler(state=init_state, estimator_condition=Y, state_length=None)

        X_Hat = X_Hat.squeeze(1).permute(0, 2, 1)

        return X_Hat



class ConditionalFlow(ABC):
    """
    Abstract class for different conditional flow-matching (CFM) classes

    Time horizon is [time_min, time_max (should be 1)]

    every path is "conditioned" on endpoints of the path
    endpoints are just our paired data samples
    subclasses need to implement mean, std, and vector_field

    """

    def __init__(self, time_min: float = 1e-8, time_max: float = 1.0):
        self.time_min = time_min
        self.time_max = time_max

    @abstractmethod
    def mean(self, *, time: torch.Tensor, x_start: torch.Tensor, x_end: torch.Tensor) -> torch.Tensor:
        """
        Return the mean of p_t(x | x_start, x_end) at time t
        """
        pass

    @abstractmethod
    def std(self, *, time: torch.Tensor, x_start: torch.Tensor, x_end: torch.Tensor) -> torch.Tensor:
        """
        Return the standard deviation of p_t(x | x_start, x_end) at time t
        """
        pass

    @abstractmethod
    def vector_field(
        self, *, time: torch.Tensor, x_start: torch.Tensor, x_end: torch.Tensor, point: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the conditional vector field v_t( point | x_start, x_end)
        """
        pass

    @staticmethod
    def _broadcast_time(time: torch.Tensor, n_dim: int) -> torch.Tensor:
        """
        Broadcast time tensor to the desired number of dimensions
        """
        if time.ndim == 1:
            target_shape = ' '.join(['B'] + ['1'] * (n_dim - 1))
            time = einops.rearrange(time, f'B -> {target_shape}')

        return time

    def generate_time(self, batch_size: int) -> torch.Tensor:
        """
        Randomly sample a batchsize of time_steps from U[0~1]
        """
        return torch.clamp(torch.rand((batch_size,)), self.time_min, self.time_max)

    def sample(self, *, time: torch.Tensor, x_start: torch.Tensor, x_end: torch.Tensor) -> torch.Tensor:
        """
        Generate a sample from p_t(x | x_start, x_end) at time t.
        Note that this implementation assumes all path marginals are normally distributed.
        """
        time = self._broadcast_time(time, n_dim=x_start.ndim)

        mean = self.mean(time=time, x_start=x_start, x_end=x_end)
        std = self.std(time=time, x_start=x_start, x_end=x_end)
        return mean + std * torch.randn_like(mean)

    def flow(
        self, *, time: torch.Tensor, x_start: torch.Tensor, x_end: torch.Tensor, point: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the conditional flow phi_t( point | x_start, x_end).
        This is an affine flow.
        """
        mean = self.mean(time=time, x_start=x_start, x_end=x_end)
        std = self.std(time=time, x_start=x_start, x_end=x_end)
        return mean + std * (point - x_start)


class OptimalTransportFlow(ConditionalFlow):
    """The OT-CFM model from [Lipman et at, 2023]

    Every conditional path the following holds:
    p_0 = N(x_start, sigma_start)
    p_1 = N(x_end, sigma_end),

    mean(x, t) = (time_max - t) * x_start + t * x_end
        (linear interpolation between x_start and x_end)

    std(x, t) = (time_max - t) * sigma_start + t * sigma_end

    Every conditional path is optimal transport map from p_0(x_start, x_end) to p_1(x_start, x_end)
    Marginal path is not guaranteed to be an optimal transport map from p_0 to p_1

    To get the OT-CFM model from [Lipman et at, 2023] just pass zeroes for x_start
    To get the I-CFM model, set sigma_min=sigma_max
    To get the rectified flow model, set sigma_min=sigma_max=0

    Args:
        time_min: minimum time value used in the process
        time_max: maximum time value used in the process
        sigma_start: the standard deviation of the initial distribution
        sigma_end: the standard deviation of the target distribution
    """

    def __init__(
        self, time_min: float = 1e-8, time_max: float = 1.0, sigma_start: float = 1.0, sigma_end: float = 1e-4
    ):
        super().__init__(time_min=time_min, time_max=time_max)
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end

    def mean(self, *, x_start: torch.Tensor, x_end: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return (self.time_max - time) * x_start + time * x_end

    def std(self, *, x_start: torch.Tensor, x_end: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return (self.time_max - time) * self.sigma_start + time * self.sigma_end

    def vector_field(
        self,
        *,
        x_start: torch.Tensor,
        x_end: torch.Tensor,
        time: torch.Tensor,
        point: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        time = self._broadcast_time(time, n_dim=x_start.ndim)

        if self.sigma_start == self.sigma_end:
            return x_end - x_start

        num = self.sigma_end * (point - x_start) - self.sigma_start * (point - x_end)
        denom = (1 - time) * self.sigma_start + time * self.sigma_end
        return num / (denom + eps)


class ConditionalFlowMatchingSampler(ABC):
    """
    Abstract class for different sampler to solve the ODE in CFM

    Args:
        estimator: the NN-based conditional vector field estimator
        num_steps: How many time steps to iterate in the process
        time_min: minimum time value used in the process
        time_max: maximum time value used in the process

    """

    def __init__(
        self,
        estimator: torch.nn.Module,
        num_steps: int = 5,
        time_min: float = 1e-8,
        time_max: float = 1.0,
    ):
        self.estimator = estimator
        self.num_steps = num_steps
        self.time_min = time_min
        self.time_max = time_max

    @property
    def time_step(self):
        return (self.time_max - self.time_min) / self.num_steps

    @abstractmethod
    def forward(
        self, state: torch.Tensor, estimator_condition: torch.Tensor, state_length: torch.Tensor):
        pass


class ConditionalFlowMatchingEulerSampler(ConditionalFlowMatchingSampler):
    """
    The Euler Sampler for solving the ODE in CFM on a uniform time grid
    """

    def __init__(
        self,
        estimator: torch.nn.Module,
        num_steps: int = 5,
        time_min: float = 1e-8,
        time_max: float = 1.0,
    ):
        super().__init__(
            estimator=estimator,
            num_steps=num_steps,
            time_min=time_min,
            time_max=time_max,
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @torch.inference_mode()
    def forward(
        self, state: torch.Tensor, estimator_condition: torch.Tensor, state_length: torch.Tensor):
        time_steps = torch.linspace(self.time_min, self.time_max, self.num_steps)

        if state_length is not None:
            state = mask_sequence_tensor(state, state_length)

        for t in time_steps:
            time = t * torch.ones(state.shape[0], device=state.device)

            if estimator_condition is None:
                estimator_input = state
            else:
                estimator_input = torch.cat([state, estimator_condition], dim=1)

            # SEMambaSeparator using ilens
            vector_field, *_ = self.estimator(input=estimator_input, ilens=state_length, time_cond=time)
            vector_field = vector_field[0].unsqueeze(1)
            # vector_field, _ = self.estimator(input=estimator_input, input_length=state_length, condition=time)

            state = state + vector_field * self.time_step

            if state_length is not None:
                state = mask_sequence_tensor(state, state_length)

        return state, state_length


def mask_sequence_tensor(tensor: torch.Tensor, lengths: torch.Tensor):
    """
    For tensors containing sequences, zero out out-of-bound elements given lengths of every element in the batch.

    tensor: tensor of shape (B, L), (B, D, L) or (B, D1, D2, L),
    lengths: LongTensor of shape (B,)
    """
    batch_size, *_, max_lengths = tensor.shape

    if len(tensor.shape) == 2:
        mask = torch.ones(batch_size, max_lengths).cumsum(dim=-1).type_as(lengths)
        mask = mask <= einops.rearrange(lengths, 'B -> B 1')
    elif len(tensor.shape) == 3:
        mask = torch.ones(batch_size, 1, max_lengths).cumsum(dim=-1).type_as(lengths)
        mask = mask <= einops.rearrange(lengths, 'B -> B 1 1')
    elif len(tensor.shape) == 4:
        mask = torch.ones(batch_size, 1, 1, max_lengths).cumsum(dim=-1).type_as(lengths)
        mask = mask <= einops.rearrange(lengths, 'B -> B 1 1 1')
    else:
        raise ValueError('Can only mask tensors of shape B x L, B x D x L and B x D1 x D2 x L')

    return tensor * mask


flow_estimator_choices = ClassChoices(
    name="flow_estimator",
    classes=dict(se_mamba=SEMambaSeparator),
    type_check=torch.nn.Module,
    default=None,
)

flow_sampler_choices = ClassChoices(
    name="flow_sampler",
    classes=dict(cfm_euler=ConditionalFlowMatchingEulerSampler),
    type_check=ConditionalFlowMatchingSampler,
    default=None,
)

flow_choices = ClassChoices(
    name="flow",
    classes=dict(optimal_transport_flow=OptimalTransportFlow),
    type_check=ConditionalFlow,
    default=None,
)