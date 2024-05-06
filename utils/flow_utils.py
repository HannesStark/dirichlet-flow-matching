import copy
import math
import pickle

import scipy
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import sqrtm


def load_flybrain_designed_seqs(path):
    order = {'A': 0, 'C':1, 'G':2, 'T':3}
    f = open(path, "rb")
    data = pickle.load(f)
    arrays = []
    for seq in data['seq']:
        arrays.append([order[char] for char in seq])
    return torch.tensor(arrays, dtype=torch.long)


def update_ema(current_dict, prev_ema, gamma = 0.9):
    ema = copy.deepcopy(prev_ema)
    current_dict = copy.deepcopy(current_dict)
    for key, current_value in current_dict.items():
        ema_key  = 'ema_' + key
        if not np.isnan(current_value):
            if ema_key in prev_ema:
                ema[ema_key] = (1 - gamma) * current_value + gamma * prev_ema[ema_key]
            else:
                ema[ema_key] = current_value
    return ema

def min_max_str(x):
    return f'min {x.min()} max {x.max()}'

def get_wasserstein_dist(embeds1, embeds2):
    if np.isnan(embeds2).any() or np.isnan(embeds1).any() or len(embeds1) == 0 or len(embeds2) == 0:
        return float('nan')
    mu1, sigma1 = embeds1.mean(axis=0), np.cov(embeds1, rowvar=False)
    mu2, sigma2 = embeds2.mean(axis=0), np.cov(embeds2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    dist = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return dist

def simplex_proj(seq):
    """Algorithm from https://arxiv.org/abs/1309.1541 Weiran Wang, Miguel Á. Carreira-Perpiñán"""
    Y = seq.reshape(-1, seq.shape[-1])
    N, K = Y.shape
    X, _ = torch.sort(Y, dim=-1, descending=True)
    X_cumsum = torch.cumsum(X, dim=-1) - 1
    div_seq = torch.arange(1, K + 1, dtype=Y.dtype, device=Y.device)
    Xtmp = X_cumsum / div_seq.unsqueeze(0)

    greater_than_Xtmp = (X > Xtmp).sum(dim=1, keepdim=True)
    row_indices = torch.arange(N, dtype=torch.long, device=Y.device).unsqueeze(1)
    selected_Xtmp = Xtmp[row_indices, greater_than_Xtmp - 1]

    X = torch.max(Y - selected_Xtmp, torch.zeros_like(Y))
    return X.view(seq.shape)



def batch_project_simplex(v):
    u, _ = torch.sort(v, dim=1, descending=True)
    cssv = u.cumsum(dim=1)
    k = torch.arange(1, v.shape[1] + 1, device=v.device)
    rho = ((u * k) > (cssv - 1)).int().cumsum(dim=1).argmax(dim=1)
    theta = (cssv[torch.arange(v.shape[0]), rho] - 1) / (rho + 1).float()
    w = torch.maximum(v - theta.unsqueeze(1), torch.tensor(0.0, device=v.device))
    return w

if __name__ == "__main__":
    a = torch.softmax(torch.rand((5,4)), dim=-1)
    b = torch.rand((5,4)) - 1
    ab = torch.cat([a,b])
    ab_proj1 = batch_project_simplex(ab)
    ab_proj2 = simplex_proj(ab)
    print('ab_proj1 - ab_proj2',ab_proj1 - ab_proj2)
    print('ab_proj1 - ab', ab_proj1 - ab)
    print('ab_proj2.sum(-1)', ab_proj2.sum(-1))
    print('ab_proj2', ab_proj2)

def sample_cond_prob_path(args, seq, alphabet_size):
    B, L = seq.shape
    seq_one_hot = torch.nn.functional.one_hot(seq, num_classes=alphabet_size)
    if args.mode == 'dirichlet':
        alphas = torch.from_numpy(1 + scipy.stats.expon().rvs(size=B) * args.alpha_scale).to(seq.device).float()
        if args.fix_alpha:
            alphas = torch.ones(B, device=seq.device) * args.fix_alpha
        alphas_ = torch.ones(B, L, alphabet_size, device=seq.device)
        alphas_ = alphas_ + seq_one_hot * (alphas[:,None,None] - 1)
        xt = torch.distributions.Dirichlet(alphas_).sample()
    elif args.mode == 'distill':
        alphas = torch.zeros(B, device=seq.device)
        xt = torch.distributions.Dirichlet(torch.ones(B, L, alphabet_size, device=seq.device)).sample()
    elif args.mode == 'riemannian':
        t = torch.rand(B, device=seq.device)
        dirichlet = torch.distributions.Dirichlet(torch.ones(alphabet_size, device=seq.device))
        x0 = dirichlet.sample((B,L))
        x1 = seq_one_hot
        xt = t[:,None,None] * x1 + (1 - t[:,None,None]) * x0
        alphas = t
    elif args.mode == 'ardm' or args.mode == 'lrar':
        mask_prob = torch.rand(1, device=seq.device)
        mask = torch.rand(seq.shape, device=seq.device) < mask_prob
        if args.mode == 'lrar': mask = ~(torch.arange(L, device=seq.device) < (1-mask_prob) * L)
        xt = torch.where(mask, alphabet_size, seq) # mask token index
        xt = torch.nn.functional.one_hot(xt, num_classes=alphabet_size + 1).float() # plus one to include index for mask token
        alphas = mask_prob.expand(B)
    return xt, alphas

def expand_simplex(xt, alphas, prior_pseudocount):
    prior_weights = (prior_pseudocount / (alphas + prior_pseudocount - 1))[:, None, None]
    return torch.cat([xt * (1 - prior_weights), xt * prior_weights], -1), prior_weights


class DirichletConditionalFlow:
    def __init__(self, K=20, alpha_min=1, alpha_max=100, alpha_spacing=0.01):
        self.alphas = np.arange(alpha_min, alpha_max + alpha_spacing, alpha_spacing)
        self.beta_cdfs = []
        self.bs = np.linspace(0, 1, 1000)
        for alph in self.alphas:
            self.beta_cdfs.append(scipy.special.betainc(alph, K-1, self.bs))
        self.beta_cdfs = np.array(self.beta_cdfs)
        self.beta_cdfs_derivative = np.diff(self.beta_cdfs, axis=0) / alpha_spacing
        self.K = K

    def c_factor(self, bs, alpha):
        out1 = scipy.special.beta(alpha, self.K - 1)
        out2 = np.where(bs < 1, out1 / ((1 - bs) ** (self.K - 1)), 0)
        out = np.where((bs ** (alpha - 1)) > 0, out2 / (bs ** (alpha - 1)), 0)
        I_func = self.beta_cdfs_derivative[np.argmin(np.abs(alpha - self.alphas))]
        interp = -np.interp(bs, self.bs, I_func)
        final = interp * out
        return final


class GaussianSmearing(torch.nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, embedding_dim=50):
        super().__init__()
        offset = torch.linspace(start, stop, embedding_dim)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)
        self.embedding_dim = embedding_dim

    def forward(self, signal):
        shape = signal.shape
        signal = signal.view(-1, 1) - self.offset.view(1, -1) + 1E-6
        encoded = torch.exp(self.coeff * torch.pow(signal, 2))
        return encoded.view(*shape, self.embedding_dim)


class MonotonicFunction(torch.nn.Module):
    def __init__(self, init_max, num_bins):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(num_bins) * np.log(init_max) - np.log(num_bins))
        self.num_bins = num_bins

    def forward(self, t):
        widths = torch.exp(self.w)
        right = torch.cumsum(widths, 0)
        left = right - widths

        bin_idx = (t * self.num_bins).long()
        frac_part = t - bin_idx * (1 / self.num_bins)

        return left[bin_idx] + (frac_part * self.num_bins) * (right[bin_idx] - left[bin_idx])

    def invert(self, f):
        widths = torch.exp(self.w)
        left = torch.cumsum(widths, 0) - widths
        bin_idx = (f.unsqueeze(-1) > left).sum(-1) - 1
        frac_part = f - left[bin_idx]
        return bin_idx / self.num_bins + frac_part / widths[bin_idx] / self.num_bins

    def derivative(self, t):
        widths = torch.exp(self.w)
        right = torch.cumsum(widths, 0)
        left = right - widths
        bin_idx = (t * self.num_bins).long()
        return (right[bin_idx] - left[bin_idx]) * self.num_bins

class SinusoidalEmbedding(nn.Module):
    """ from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py   """
    def __init__(self, embedding_dim, embedding_scale, max_positions=10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_positions = max_positions
        self.embedding_scale = embedding_scale

    def forward(self, signal):
        shape = signal.shape
        signal = signal.view(-1) * self.embedding_scale
        half_dim = self.embedding_dim // 2
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=signal.device) * -emb)
        emb = signal.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode='constant')
        assert emb.shape == (signal.shape[0], self.embedding_dim)
        return emb.view(*shape, self.embedding_dim )


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels.
    from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32
    """

    def __init__(self, embedding_dim=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_dim//2) * scale, requires_grad=False)
        self.embedding_dim = embedding_dim

    def forward(self, signal):
        shape = signal.shape
        signal = signal.view(-1)
        signal_proj = signal[:, None] * self.W[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(signal_proj), torch.cos(signal_proj)], dim=-1)
        return emb.view(*shape, self.embedding_dim )

def get_signal_mapping(embedding_type, embedding_dim, embedding_scale=10000):
    if embedding_type == 'sinusoidal':
        emb_func = SinusoidalEmbedding(embedding_dim=embedding_dim, embedding_scale=embedding_scale)
    elif embedding_type == 'fourier':
        emb_func = GaussianFourierProjection(embedding_dim=embedding_dim, scale=embedding_scale)
    elif embedding_type == 'gaussian':
        emb_func = GaussianSmearing(0.0, 1, embedding_dim)
    else:
        raise NotImplemented
    return emb_func

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def get_beta_schedule(num_steps):

    return betas_for_alpha_bar(
            num_steps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )


class GaussianDiffusionSchedule:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
            self,
            timesteps,
            noise_scale=1.0,
    ):
        betas = get_beta_schedule(timesteps)

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.timesteps = int(betas.shape[0])
        self.noise_scale = noise_scale

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = self.noise_scale * torch.randn_like(x_start)
            # add scaling here
        assert noise.shape == x_start.shape
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )

        posterior_variance = (self.noise_scale ** 2) * posterior_variance
        posterior_log_variance_clipped = 2 * np.log(self.noise_scale) + posterior_log_variance_clipped

        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding