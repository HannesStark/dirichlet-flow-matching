import copy

import numpy as np
import pandas as pd
import torch, time, os
import wandb
import yaml
from selene_sdk.utils import NonStrandSpecific
from torch import optim

from utils.esm import upgrade_state_dict
from utils.flow_utils import DirichletConditionalFlow, expand_simplex, sample_cond_prob_path, simplex_proj
from lightning_modules.general_module import GeneralModule
from utils.logging import get_logger
from model.promoter_model import PromoterModel
from utils.sei import Sei

logger = get_logger(__name__)


class PromoterModule(GeneralModule):
    def __init__(self, args):
        super().__init__(args)

        self.model = PromoterModel(args)
        self.condflow = DirichletConditionalFlow(K=self.model.alphabet_size, alpha_spacing=0.01, alpha_max=args.alpha_max)

        self.seifeatures = pd.read_csv('data/promoter_design/target.sei.names', sep='|', header=None)
        self.sei_cache = {}
        self.loaded_distill_model = False

    def on_load_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = {k: v for k,v in checkpoint['state_dict'].items() if 'distill_model' not in k}

    def training_step(self, batch, batch_idx):
        self.stage = 'train'
        loss = self.general_step(batch, batch_idx)
        if self.args.ckpt_iterations is not None and self.trainer.global_step in self.args.ckpt_iterations:
            self.trainer.save_checkpoint(os.path.join(os.environ["MODEL_DIR"],f"epoch={self.trainer.current_epoch}-step={self.trainer.global_step}.ckpt"))
        return loss

    def validation_step(self, batch, batch_idx):
        self.stage = 'val'
        loss = self.general_step(batch, batch_idx)
        if self.args.validate:
            self.try_print_log()

    def general_step(self, batch, batch_idx=None):
        self.iter_step += 1
        seq_one_hot = batch[:, :, :4]
        seq = torch.argmax(seq_one_hot, dim=-1)
        signal = batch[:, :, 4:5]
        
        B, L = seq.shape

        if self.args.mode == 'dirichlet' or self.args.mode == 'riemannian':
            xt, alphas = sample_cond_prob_path(self.args, seq, self.model.alphabet_size)
            xt, prior_weights = expand_simplex(xt, alphas, self.args.prior_pseudocount)
            self.lg('prior_weight', prior_weights)
        elif self.args.mode == 'ardm' or self.args.mode == 'lrar':
            mask_prob = torch.rand(1, device=self.device)
            mask = torch.rand(seq.shape, device=self.device) < mask_prob
            if self.args.mode == 'lrar': mask = ~(torch.arange(L, device=self.device) < (1-mask_prob) * L)
            xt = torch.where(mask, 4, seq) # mask token has idx 4
            xt = torch.nn.functional.one_hot(xt, num_classes=5)
            alphas = mask_prob.expand(B)
        elif self.args.mode == 'distill':
            if self.stage == 'val':
                seq_distill = torch.zeros_like(seq, device=self.device)
                xt = torch.ones((B,L, self.model.alphabet_size), device=self.device)
                xt = xt / xt.sum(-1)[..., None]
            else:
                logits_distill, xt = self.dirichlet_flow_inference(seq, signal, model=self.distill_model, args=self.distill_args)
                seq_distill = torch.argmax(logits_distill, dim=-1)
            alphas = torch.zeros(B, device=self.device)

        logits = self.model(xt, signal=signal, t=alphas)

        losses = torch.nn.functional.cross_entropy(logits.transpose(1, 2), seq_distill if self.args.mode == 'distill' else seq, reduction='none')
        losses = losses.mean(-1)

        self.lg('alpha', alphas)
        self.lg('loss', losses)
        self.lg('perplexity', torch.exp(losses.mean())[None].expand(B))
        self.lg('dur', torch.tensor(time.time() - self.last_log_time)[None].expand(B))
        if self.stage == "val":

            if self.args.mode == 'dirichlet':
                logits_pred, _ = self.dirichlet_flow_inference(seq, signal, self.model, args=self.args)
                seq_pred = torch.argmax(logits_pred, dim=-1)
            elif self.args.mode == 'riemannian':
                logits_pred = self.riemannian_flow_inference(seq, signal)
                seq_pred = torch.argmax(logits_pred, dim=-1)
            elif self.args.mode == 'ardm' or self.args.mode == 'lrar':
                seq_pred = self.ar_inference(seq, signal)
            elif self.args.mode == 'distill':
                logits_pred = self.distill_inference(seq, signal)
                seq_pred = torch.argmax(logits_pred, dim=-1)
            else:
                raise NotImplementedError()
            self.lg('seq', [''.join([['A','C','G','T'][num] for num in seq]) for seq in seq_pred])
            seq_pred_one_hot = torch.nn.functional.one_hot(seq_pred, num_classes=self.model.alphabet_size).float()

            if batch_idx not in self.sei_cache:
                sei_profile = self.get_sei_profile(seq_one_hot)
                self.sei_cache = sei_profile
            else:
                sei_profile = self.sei_cache[batch_idx]

            sei_profile_pred = self.get_sei_profile(seq_pred_one_hot)
            self.lg('sp-mse', ((sei_profile - sei_profile_pred) ** 2))
            self.lg('recovery', seq_pred.eq(seq).float().mean(-1))

        self.last_log_time = time.time()
        return losses.mean()

    def get_sei_profile(self, seq_one_hot):
        B, L, K = seq_one_hot.shape
        sei_inp = torch.cat([torch.ones((B, 4, 1536), device=self.device) * 0.25,
                             seq_one_hot.transpose(1, 2),
                             torch.ones((B, 4, 1536), device=self.device) * 0.25], 2) # batchsize x 4 x 4,096
        sei_out = self.sei(sei_inp).cpu().detach().numpy() # batchsize x 21,907
        sei_out = sei_out[:, self.seifeatures[1].str.strip().values == 'H3K4me3'] # batchsize x 2,350
        predh3k4me3 = sei_out.mean(axis=1) # batchsize
        return predh3k4me3

    @torch.no_grad()
    def distill_inference(self, seq,signal):
        B, L = seq.shape
        K = self.model.alphabet_size
        x0 = torch.distributions.Dirichlet(torch.ones(B, L, K, device=seq.device)).sample()
        logits = self.model(x0,signal, t=torch.zeros(B, device=self.device))
        return logits

    @torch.no_grad()
    def riemannian_flow_inference(self, seq, signal, batch_idx=None):
        B, L = seq.shape
        K = self.model.alphabet_size
        xt = torch.distributions.Dirichlet(torch.ones(B, L, K)).sample().to(self.device)
        eye = torch.eye(K).to(self.device)

        t_span = torch.linspace(0, 1, self.args.num_integration_steps, device=self.device)
        for s, t in zip(t_span[:-1], t_span[1:]):
            xt_expanded, prior_weights = expand_simplex(xt, s[None].expand(B), self.args.prior_pseudocount)
            logits = self.model(xt_expanded, signal, s[None].expand(B))
            probs = torch.nn.functional.softmax(logits, -1)
            cond_flows = (eye - xt.unsqueeze(-1)) / (1 - s)
            flow = (probs.unsqueeze(-2) * cond_flows).sum(-1)
            xt = xt + flow * (t - s)

        return logits

    @torch.no_grad()
    def dirichlet_flow_inference(self, seq, signal, model, args):
        B, L = seq.shape
        x0 = torch.distributions.Dirichlet(torch.ones(B, L, model.alphabet_size, device=seq.device)).sample()
        eye = torch.eye(model.alphabet_size).to(x0)
        xt = x0

        t_span = torch.linspace(1, args.alpha_max, self.args.num_integration_steps, device=self.device)
        for i, (s, t) in enumerate(zip(t_span[:-1], t_span[1:])):
            prior_weight = args.prior_pseudocount / (s + args.prior_pseudocount - 1)
            seq_xt = torch.cat([xt * (1 - prior_weight), xt * prior_weight], -1)

            logits = model(seq_xt, signal, s[None].expand(B))
            out_probs = torch.nn.functional.softmax(logits / args.flow_temp, -1)

            c_factor = self.condflow.c_factor(xt.cpu().numpy(), s.item())
            c_factor = torch.from_numpy(c_factor).to(xt)
            if torch.isnan(c_factor).any():
                print(f'NAN cfactor after: xt.min(): {xt.min()}, out_probs.min(): {out_probs.min()}')
                c_factor = torch.nan_to_num(c_factor)

            cond_flows = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)
            flow = (out_probs.unsqueeze(-2) * cond_flows).sum(-1)
            xt = xt + flow * (t - s)
            if not torch.allclose(xt.sum(2), torch.ones((B, L), device=self.device), atol=1e-4) or not (xt >= 0).all():
                print(f'WARNING: xt.min(): {xt.min()}. Some values of xt do not lie on the simplex. There are we are {(xt<0).sum()} negative values in xt of shape {xt.shape} that are negative.')
                xt = simplex_proj(xt)

        return logits, x0


    @torch.no_grad()
    def ar_inference(self, seq, signal):
        B, L = seq.shape
        order = np.arange(L)
        if self.args.mode =='ardm': np.random.shuffle(order)
        curr = (torch.ones((B, L), device=self.device) * 4).long()
        for i, k in enumerate(order):
            t = torch.tensor( i/L ,device=self.device)
            logits = self.model(torch.nn.functional.one_hot(curr, num_classes=5), signal, t[None].expand(B))
            curr[:, k] = torch.distributions.Categorical(probs=torch.nn.functional.softmax(logits[:, k] / self.args.flow_temp, -1)).sample()
        return curr

    @torch.no_grad()
    def on_validation_epoch_start(self) -> None:
        logger.info('Loading sei model')
        self.sei = NonStrandSpecific(Sei(4096, 21907))
        self.sei.load_state_dict(upgrade_state_dict(
            torch.load('data/promoter_design/best.sei.model.pth.tar', map_location='cpu')['state_dict'],
            prefixes=['module.']))
        self.sei.to(self.device)
        self.generator = np.random.default_rng(seed=137)

    def load_distill_model(self):
        with open(self.args.distill_ckpt_hparams) as f:
            hparams = yaml.load(f, Loader=yaml.UnsafeLoader)
            self.distill_args = copy.deepcopy(hparams['args'])
        self.distill_model =  PromoterModel(self.distill_args)
        upgraded_dict = upgrade_state_dict(torch.load(self.args.distill_ckpt, map_location=self.device)['state_dict'], prefixes=['model.'])
        self.distill_model.load_state_dict(upgraded_dict)
        self.distill_model.eval()
        self.distill_model.to(self.device)
        for param in self.distill_model.parameters():
            param.requires_grad = False


    def on_train_epoch_start(self) -> None:
        if not self.loaded_distill_model and self.args.distill_ckpt is not None:
            self.load_distill_model()
            self.loaded_distill_model = True

    def on_validation_epoch_end(self):
        del self.sei
        torch.cuda.empty_cache()
        self.generator = np.random.default_rng()
        log = self._log
        log = {key: log[key] for key in log if "val_" in key}
        log = self.gather_log(log, self.trainer.world_size)
        mean_log = self.get_log_mean(log)
        mean_log.update({'epoch': self.trainer.current_epoch, 'step': self.trainer.global_step, 'iter_step': self.iter_step})

        if self.trainer.is_global_zero:
            logger.info(str(mean_log))
            self.log_dict(mean_log, batch_size=1)
            if self.args.wandb:
                wandb.log(mean_log)

            path = os.path.join(os.environ["MODEL_DIR"], f"val_{self.trainer.global_step}.csv")
            pd.DataFrame(log).to_csv(path)

        for key in list(log.keys()):
            if "val_" in key:
                del self._log[key]

    def lg(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        log = self._log
        if self.args.validate or self.stage == 'train':
            log["iter_" + key].extend(data)
        log[self.stage + "_" + key].extend(data)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer

