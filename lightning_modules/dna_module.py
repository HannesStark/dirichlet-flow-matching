import copy
import math
from collections import defaultdict

import PIL
import numpy as np
import pandas as pd
import torch, time, os
import wandb
import seaborn as sns
import yaml

sns.set_style('whitegrid')
from matplotlib import pyplot as plt
from torch import optim

from model.dna_models import MLPModel, CNNModel, TransformerModel, DeepFlyBrainModel
from utils.esm import upgrade_state_dict
from utils.flow_utils import DirichletConditionalFlow, expand_simplex, sample_cond_prob_path, simplex_proj, \
    get_wasserstein_dist, update_ema, load_flybrain_designed_seqs
from lightning_modules.general_module import GeneralModule
from utils.logging import get_logger


logger = get_logger(__name__)


class DNAModule(GeneralModule):
    def __init__(self, args, alphabet_size, num_cls, toy_data):
        super().__init__(args)
        self.load_model(alphabet_size, num_cls)

        self.condflow = DirichletConditionalFlow(K=self.model.alphabet_size, alpha_spacing=0.001, alpha_max=args.alpha_max)
        self.crossent_loss = torch.nn.CrossEntropyLoss(reduction='none')

        self.toy_data = toy_data

        self.val_outputs = defaultdict(list)
        self.train_outputs = defaultdict(list)
        self.train_out_initialized = False
        self.loaded_classifiers = False
        self.loaded_distill_model = False
        self.mean_log_ema = {}
        if self.args.taskiran_seq_path is not None:
            self.taskiran_fly_seqs = load_flybrain_designed_seqs(self.args.taskiran_seq_path).to(self.device)


    def on_load_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = {k: v for k,v in checkpoint['state_dict'].items() if 'cls_model' not in k and 'distill_model' not in k}

    def training_step(self, batch, batch_idx):
        self.stage = 'train'
        loss = self.general_step(batch, batch_idx)
        if self.args.ckpt_iterations is not None and self.trainer.global_step in self.args.ckpt_iterations:
            self.trainer.save_checkpoint(os.path.join(os.environ["MODEL_DIR"],f"epoch={self.trainer.current_epoch}-step={self.trainer.global_step}.ckpt"))
        self.try_print_log()
        return loss

    def validation_step(self, batch, batch_idx):
        self.stage = 'val'
        loss = self.general_step(batch, batch_idx)
        if self.args.validate:
            self.try_print_log()

    def general_step(self, batch, batch_idx=None):
        self.iter_step += 1
        seq, cls = batch
        B, L = seq.shape

        xt, alphas = sample_cond_prob_path(self.args, seq, self.model.alphabet_size)
        if self.args.mode == 'distill':
            if self.stage == 'val':
                seq_distill = torch.zeros_like(seq, device=self.device)
            else:
                logits_distill, xt = self.dirichlet_flow_inference(seq, cls, model=self.distill_model, args=self.distill_args)
                seq_distill = torch.argmax(logits_distill, dim=-1)
            alphas = alphas * 0
        xt_inp = xt
        if self.args.mode == 'dirichlet' or self.args.mode == 'riemannian':
            xt_inp, prior_weights = expand_simplex(xt,alphas, self.args.prior_pseudocount)
            self.lg('prior_weight', prior_weights)

        if self.args.cls_free_guidance:
            if self.args.binary_guidance:
                cls_inp = cls.clone()
                cls_inp[cls != self.args.target_class] = self.model.num_cls
            else:
                cls_inp = torch.where(torch.rand(B, device=self.device) >= self.args.cls_free_noclass_ratio, cls.squeeze(), self.model.num_cls) # set fraction of the classes to the unconditional class
        else:
            cls_inp = None
        logits = self.model(xt_inp, t=alphas, cls=cls_inp)

        losses = torch.nn.functional.cross_entropy(logits.transpose(1, 2), seq_distill if self.args.mode == 'distill' else seq, reduction='none')
        losses = losses.mean(-1)

        self.lg('loss', losses)
        self.lg('perplexity', torch.exp(losses.mean())[None].expand(B))
        if self.stage == "val":
            if self.args.mode == 'dirichlet':
                logits_pred, _ = self.dirichlet_flow_inference(seq, cls, model=self.model, args=self.args)
                seq_pred = torch.argmax(logits_pred, dim=-1)
            elif self.args.mode == 'riemannian':
                logits_pred = self.riemannian_flow_inference(seq)
                seq_pred = torch.argmax(logits_pred, dim=-1)
            elif self.args.mode == 'ardm' or self.args.mode == 'lrar':
                seq_pred = self.ar_inference(seq)
            elif self.args.mode == 'distill':
                logits_pred = self.distill_inference(seq)
                seq_pred = torch.argmax(logits_pred, dim=-1)
            else:
                raise NotImplementedError()

            self.lg('seq', [''.join([['A','C','G','T'][num] if self.model.alphabet_size == 4 else str(num) for num in seq]) for seq in seq_pred])
            self.lg('recovery', seq_pred.eq(seq).float().mean(-1))
            if self.args.dataset_type == 'toy_fixed':
                self.log_data_similarities(seq_pred)

            self.val_outputs['seqs'].append(seq_pred.cpu())
            if self.args.cls_ckpt is not None:
                #self.run_cls_model(seq_pred, cls, log_dict=self.val_outputs, clean_data=False, postfix='_noisycls_generated', generated=True)
                self.run_cls_model(seq, cls, log_dict=self.val_outputs, clean_data=False, postfix='_noisycls', generated=False)
            if self.args.clean_cls_ckpt is not None:
                self.run_cls_model(seq_pred, cls, log_dict=self.val_outputs, clean_data=True, postfix='_cleancls_generated', generated=True)
                self.run_cls_model(seq, cls, log_dict=self.val_outputs, clean_data=True, postfix='_cleancls', generated=False)
                if self.args.taskiran_seq_path is not None:
                    indices = torch.randperm(len(self.taskiran_fly_seqs))[:B].to(self.device)
                    self.run_cls_model(self.taskiran_fly_seqs[indices].to(self.device), cls, log_dict=self.val_outputs, clean_data=True, postfix='_cleancls_taskiran', generated=True)
        self.lg('alpha', alphas)
        self.lg('dur', torch.tensor(time.time() - self.last_log_time)[None].expand(B))
        if not self.train_out_initialized and self.args.clean_cls_ckpt is not None:
            self.run_cls_model(seq, cls, log_dict=self.train_outputs, clean_data=True, postfix='_cleancls', generated=False, run_log=False)
        self.last_log_time = time.time()
        return losses.mean()

    @torch.no_grad()
    def distill_inference(self, seq):
        B, L = seq.shape
        K = self.model.alphabet_size
        x0 = torch.distributions.Dirichlet(torch.ones(B, L, K, device=seq.device)).sample()
        logits = self.model(x0, t=torch.zeros(B, device=self.device))
        return logits

    @torch.no_grad()
    def dirichlet_flow_inference(self, seq, cls, model, args):
        B, L = seq.shape
        K = model.alphabet_size
        x0 = torch.distributions.Dirichlet(torch.ones(B, L, model.alphabet_size, device=seq.device)).sample()
        eye = torch.eye(K).to(x0)
        xt = x0.clone()

        t_span = torch.linspace(1, args.alpha_max, self.args.num_integration_steps, device=self.device)
        for i, (s, t) in enumerate(zip(t_span[:-1], t_span[1:])):
            xt_expanded, prior_weights = expand_simplex(xt, s[None].expand(B), args.prior_pseudocount)
            if args.cls_free_guidance:
                logits = model(xt_expanded, t=s[None].expand(B), cls=cls if self.args.all_class_inference else (torch.ones(B, device=self.device) * args.target_class).long())
                probs_cond = torch.nn.functional.softmax(logits / args.flow_temp, -1)  # [B, L, K]
                if self.args.score_free_guidance:
                    flow_probs = probs_cond
                else:
                    logits_uncond = model(xt_expanded, t=s[None].expand(B), cls=(torch.ones(B, device=self.device) * model.num_cls).long())
                    probs_unccond = torch.nn.functional.softmax(logits_uncond / args.flow_temp, -1)  # [B, L, K]
                    if self.args.probability_tilt:
                        flow_probs = probs_cond ** (1 - self.args.guidance_scale) * probs_unccond ** (self.args.guidance_scale)
                        flow_probs = flow_probs / flow_probs.sum(-1)[...,None]
                    elif self.args.probability_addition:
                        if self.args.adaptive_prob_add:
                            #TODO this is wrong for some reason and we get negative probabilities ?!?!??!
                            potential_scales = probs_cond / (probs_cond - probs_unccond)
                            max_guide_scale = potential_scales.min(-1)[0]
                            flow_probs = probs_cond * (1 - max_guide_scale[...,None]) + probs_unccond * max_guide_scale[...,None]
                        else:
                            flow_probs = probs_cond * self.args.guidance_scale + probs_unccond *(1 - self.args.guidance_scale)
                    else:
                        flow_probs = self.get_cls_free_guided_flow(xt, s + 1e-4, logits_uncond, logits)

            else:
                logits = model(xt_expanded, t=s[None].expand(B))
                flow_probs = torch.nn.functional.softmax(logits / args.flow_temp, -1) # [B, L, K]

            if args.cls_guidance:
                probs_cond, cls_score = self.get_cls_guided_flow(xt, s + 1e-4, flow_probs)
                flow_probs = probs_cond * self.args.guidance_scale + flow_probs * (1 - self.args.guidance_scale)

            if not torch.allclose(flow_probs.sum(2), torch.ones((B, L), device=self.device), atol=1e-4) or not (flow_probs >= 0).all():
                print(f'WARNING: flow_probs.min(): {flow_probs.min()}. Some values of flow_probs do not lie on the simplex. There are we are {(flow_probs<0).sum()} negative values in flow_probs of shape {flow_probs.shape} that are negative. We are projecting them onto the simplex.')
                flow_probs = simplex_proj(flow_probs)

            c_factor = self.condflow.c_factor(xt.cpu().numpy(), s.item())
            c_factor = torch.from_numpy(c_factor).to(xt)

            self.inf_counter += 1
            if torch.isnan(c_factor).any():
                print(f'NAN cfactor after: xt.min(): {xt.min()}, flow_probs.min(): {flow_probs.min()}')
                if self.args.allow_nan_cfactor:
                    c_factor = torch.nan_to_num(c_factor)
                    self.nan_inf_counter += 1
                else:
                    raise RuntimeError(f'NAN cfactor after: xt.min(): {xt.min()}, flow_probs.min(): {flow_probs.min()}')

            if not (flow_probs >= 0).all(): print(f'flow_probs.min(): {flow_probs.min()}')
            cond_flows = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)
            flow = (flow_probs.unsqueeze(-2) * cond_flows).sum(-1)


            if self.args.vectorfield_addition:
                flow_cond = (probs_cond.unsqueeze(-2) * cond_flows).sum(-1)
                flow_uncond = (probs_unccond.unsqueeze(-2) * cond_flows).sum(-1)
                flow = flow_cond * self.args.guidance_scale + (1 - self.args.guidance_scale) * flow_uncond

            xt = xt + flow * (t - s)

            if not torch.allclose(xt.sum(2), torch.ones((B, L), device=self.device), atol=1e-4) or not (xt >= 0).all():
                print(f'WARNING: xt.min(): {xt.min()}. Some values of xt do not lie on the simplex. There are we are {(xt<0).sum()} negative values in xt of shape {xt.shape} that are negative. We are projecting them onto the simplex.')
                xt = simplex_proj(xt)
        return logits, x0

    @torch.no_grad()
    def riemannian_flow_inference(self, seq):
        B, L = seq.shape
        K = self.model.alphabet_size
        xt = torch.distributions.Dirichlet(torch.ones(B, L, K)).sample().to(self.device)
        eye = torch.eye(K).to(self.device)

        t_span = torch.linspace(0, 1, self.args.num_integration_steps, device=self.device)
        for s, t in zip(t_span[:-1], t_span[1:]):
            xt_expanded, prior_weights = expand_simplex(xt, s[None].expand(B), self.args.prior_pseudocount)
            logits = self.model(xt_expanded, s[None].expand(B))
            probs = torch.nn.functional.softmax(logits, -1)
            cond_flows = (eye - xt.unsqueeze(-1)) / (1 - s)
            flow = (probs.unsqueeze(-2) * cond_flows).sum(-1)
            xt = xt + flow * (t - s)
        return xt

    @torch.no_grad()
    def ar_inference(self, seq):
        B, L = seq.shape
        order = np.arange(L)
        if self.args.mode == 'ardm': np.random.shuffle(order)
        curr = (torch.ones((B, L), device=self.device) * 4).long()
        for i, k in enumerate(order):
            t = torch.tensor(i / L, device=self.device)
            logits = self.model(torch.nn.functional.one_hot(curr, num_classes=5).float(), t[None].expand(B))
            curr[:, k] = torch.distributions.Categorical(
                probs=torch.nn.functional.softmax(logits[:, k] / self.args.flow_temp, -1)).sample()
        return curr

    def get_cls_free_guided_flow(self, xt, alpha, logits, logits_cond,):
        B, L, K = xt.shape
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs_cond = torch.nn.functional.softmax(logits_cond, dim=-1)


        cond_scores_mats = ((alpha - 1) * (torch.eye(self.model.alphabet_size).to(xt)[None, :] / xt[..., None]))  # [B, L, K, K]

        cond_scores_mats = cond_scores_mats - cond_scores_mats.mean(2)[:, :, None, :]  # [B, L, K, K] now the columns sum up to 0

        score = torch.einsum('ijkl,ijl->ijk', cond_scores_mats, probs)  # [B, L, K] add up the columns of conditional flow scores weighted by the predicted probability of each corner
        score_cond = torch.einsum('ijkl,ijl->ijk', cond_scores_mats, probs_cond)  # [B, L, K] add up the columns of conditional flow scores weighted by the predicted probability of each corner
        score_guided = (1 - self.args.guidance_scale) * score + self.args.guidance_scale * score_cond

        Q_mats = cond_scores_mats.clone()  # [B, L, K, K]
        Q_mats[:, :, -1, :] = torch.ones((B, L, K))  # [B, L, K, K]
        score_guided_ = score_guided.clone()  # [B, L, K]
        score_guided_[:, :, -1] = torch.ones(B, L)  # [B, L, K]
        flow_guided = torch.linalg.solve(Q_mats, score_guided_)  # [B, L, K]
        return flow_guided

    def get_cls_guided_flow(self, xt, alpha, p_x0_given_xt):
        B, L, K = xt.shape
        # get the matrix of scores of the conditional probability flows for each simplex corner
        cond_scores_mats = ((alpha - 1) * (torch.eye(self.model.alphabet_size).to(xt)[None, :] / xt[..., None]))  # [B, L, K, K]
        cond_scores_mats = cond_scores_mats - cond_scores_mats.mean(2)[:, :, None, :]  # [B, L, K, K] now the columns sum up to 0
        # assert torch.allclose(cond_scores_mats.sum(2), torch.zeros((B, L, K)),atol=1e-4), cond_scores_mats.sum(2)

        score = torch.einsum('ijkl,ijl->ijk', cond_scores_mats, p_x0_given_xt)  # [B, L, K] add up the columns of conditional flow scores weighted by the predicted probability of each corner
        # assert torch.allclose(score.sum(2), torch.zeros((B, L)),atol=1e-4)

        cls_score = self.get_cls_score(xt, alpha[None].expand(B))
        if self.args.scale_cls_score:
            norm_score = torch.norm(score, dim=2, keepdim=True)
            norm_cls_score = torch.norm(cls_score, dim=2, keepdim=True)
            cls_score = torch.where(norm_cls_score != 0, cls_score * norm_score / norm_cls_score, cls_score)
        guided_score = cls_score + score

        Q_mats = cond_scores_mats.clone()  # [B, L, K, K]
        Q_mats[:, :, -1, :] = torch.ones((B, L, K))  # [B, L, K, K]
        guided_score_ = guided_score.clone()  # [B, L, K]
        guided_score_[:, :, -1] = torch.ones(B, L)  # [B, L, K]
        p_x0_given_xt_y = torch.linalg.solve(Q_mats, guided_score_) # [B, L, K]
        """
        # for debugging whether these probabilities also have negative entries and are off of the simplex in other ways
        cls_score_ = cls_score.clone()  # [B, L, K]
        cls_score_[:, :, -1] = torch.ones(B, L)  # [B, L, K]
        p_xt_given_y = torch.linalg.solve(Q_mats, cls_score_)

        score_guided_ = score.clone()  # [B, L, K]
        score_guided_[:, :, -1] = torch.ones(B, L)  # [B, L, K]
        p_x0_given_xt_back = torch.linalg.solve(Q_mats, score_guided_)
        """
        if torch.isnan(p_x0_given_xt_y).any():
            print("Warning: there were this many nans in the probs_cond of the classifier score: ", torch.isnan(p_x0_given_xt_y).sum(), "We are setting them to 0.")
            p_x0_given_xt_y = torch.nan_to_num(p_x0_given_xt_y)
        return p_x0_given_xt_y, cls_score
    def get_cls_score(self, xt, alpha):
        with torch.enable_grad():
            xt_ = xt.clone().detach().requires_grad_(True)
            xt_.requires_grad = True
            if self.args.cls_expanded_simplex:
                xt_, prior_weights = expand_simplex(xt, alpha[None].expand(xt_.shape[0]), self.args.prior_pseudocount)
            if self.args.analytic_cls_score:
                assert self.args.dataset_type == 'toy_fixed', 'analytic_cls_score can only be calculated for fixed dataset'
                B, L, K = xt.shape

                x0_given_y = self.toy_data.data_class1.to(self.device) # num_seq, L
                x0_given_y_expanded = x0_given_y.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1,-1) # B, num_seq, L, 1
                xt_expanded = xt_.unsqueeze(1).expand(-1,x0_given_y_expanded.shape[1], -1, -1) # B, num_seq, L, K
                selected_xt = torch.gather(xt_expanded, dim=3, index=x0_given_y_expanded).squeeze() # [B, num_seq, L] where the indices of cls1_data were used to select entries in the K dimension
                p_xt_given_x0_y = selected_xt ** (alpha[:,None,None] - 1) # [B, num_seq, L]
                p_xt_given_y = p_xt_given_x0_y.mean(1)  # [B, L] because the probs for each x0 are the same

                x0_all = torch.cat([self.toy_data.data_class1, self.toy_data.data_class2], dim= 0).to(self.device)  # num_seq * 2, L
                x0_expanded = x0_all.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1)  # B, num_seq*2, L, 1
                xt_expanded = xt_.unsqueeze(1).expand(-1, x0_expanded.shape[1], -1, -1)  # B, num_seq*2, L, K
                selected_xt = torch.gather(xt_expanded, dim=3,index=x0_expanded).squeeze()  # [B, num_seq, L] where the indices of cls1_data were used to select entries in the K dimension
                p_xt_given_x0 = selected_xt ** (alpha[:, None, None] - 1)  # [B, num_seq, L]
                p_xt = p_xt_given_x0.mean(1)  # [B, L] because the probs for each x0 are the same

                p_y_given_xt = p_xt_given_y / 2 / p_xt # [B,L] divide by 2 becaue that is p(y) (but it does not really matter because it is just a constant)
                p_y_given_xt = p_y_given_xt.prod(-1) # per sequence probabilities. Works because positions are independent.
                log_prob = torch.log(p_y_given_xt).sum()
                cls_score = torch.autograd.grad(log_prob, [xt_])[0]
            else:
                cls_logits = self.cls_model(xt_, t=alpha)
                loss = torch.nn.functional.cross_entropy(cls_logits, torch.ones(len(xt), dtype=torch.long, device=xt.device) * self.args.target_class).mean()
                assert not torch.isnan(loss).any()
                cls_score = - torch.autograd.grad(loss,[xt_])[0]  # need the minus because cross entropy loss puts a minus in front of log probability.
                assert not torch.isnan(cls_score).any()
        cls_score = cls_score - cls_score.mean(-1)[:,:,None]
        return cls_score.detach()

    @torch.no_grad()
    def run_cls_model(self, seq, cls, log_dict, clean_data=False, postfix = '', generated=False, run_log=True):
        cls = cls.squeeze()
        if generated:
            cls = (torch.ones_like(cls,device=self.device) * self.args.target_class).long()

        xt, alphas = sample_cond_prob_path(self.args, seq, self.model.alphabet_size)
        if self.args.cls_expanded_simplex:
            xt, _ = expand_simplex(xt, alphas, self.args.prior_pseudocount)

        cls_model = self.clean_cls_model if clean_data else self.cls_model
        logits, embeddings = cls_model(xt if not clean_data else seq, t=alphas, return_embedding=True)
        cls_pred = torch.argmax(logits, dim=-1)

        if run_log:
            if not self.args.target_class == self.model.num_cls:
                losses = self.crossent_loss(logits, cls)
                self.lg(f'cls_loss{postfix}', losses)
            self.lg(f'cls_accuracy{postfix}', cls_pred.eq(cls).float())

        log_dict[f'embeddings{postfix}'].append(embeddings.detach().cpu())
        log_dict[f'clss{postfix}'].append(cls.detach().cpu())
        log_dict[f'logits{postfix}'].append(logits.detach().cpu())
        log_dict[f'alphas{postfix}'].append(alphas.detach().cpu())
        if not clean_data and not self.args.target_class == self.model.num_cls: # num_cls stands for the masked class
            scores = self.get_cls_score(xt, alphas)
            log_dict[f'scores{postfix}'].append(scores.detach().cpu())

    def on_validation_epoch_start(self):
        if not self.loaded_classifiers:
            self.load_classifiers(load_cls=self.args.cls_ckpt is not None, load_clean_cls=self.args.clean_cls_ckpt is not None)
            self.loaded_classifiers = True
        self.inf_counter = 1
        self.nan_inf_counter = 0

    def on_validation_epoch_end(self):
        self.generator = np.random.default_rng()
        log = self._log
        log = {key: log[key] for key in log if "val_" in key}
        log = self.gather_log(log, self.trainer.world_size)
        mean_log = self.get_log_mean(log)
        mean_log.update({'val_nan_inf_step_fraction': self.nan_inf_counter / self.inf_counter})

        mean_log.update({'epoch': float(self.trainer.current_epoch), 'step': float(self.trainer.global_step), 'iter_step': float(self.iter_step)})
        if self.args.dataset_type == 'toy_sampled':
            all_seqs = torch.cat(self.val_outputs['seqs'], dim=0).cpu()
            all_seqs_one_hot = torch.nn.functional.one_hot(all_seqs, num_classes=self.args.toy_simplex_dim)
            counts = all_seqs_one_hot.sum(0).float()
            empirical_dist = counts / counts.sum(dim=-1, keepdim=True)
            kl = (empirical_dist * (torch.log(empirical_dist) - torch.log(self.toy_data.probs[self.args.target_class]))).sum(-1).mean()
            rkl = (self.toy_data.probs[self.args.target_class] * (torch.log(self.toy_data.probs[self.args.target_class]) - torch.log(empirical_dist))).sum(-1).mean()
            sanity_self_kl = (empirical_dist * (torch.log(empirical_dist) - torch.log(empirical_dist))).sum(-1).mean()
            mean_log.update({'val_kl': kl.cpu().item(), 'val_rkl': rkl.cpu().item(), 'val_sanity_self_kl': sanity_self_kl.cpu().item()})

        if self.args.clean_cls_ckpt:
            if not self.args.target_class == self.model.num_cls:
                probs = torch.softmax(torch.cat(self.val_outputs['logits_cleancls_generated']), dim=-1)
                target_prob = probs[:, self.args.target_class]
                mean_log.update({'cleancls_mean_target_prob': target_prob.detach().cpu().mean()})
            # calculate FID/FXD metrics:
            embeds_gen = torch.cat(self.val_outputs['embeddings_cleancls_generated']).detach().cpu().numpy()
            if not self.args.validate:
                train_clss = torch.cat(self.train_outputs['clss_cleancls']).squeeze().detach().cpu().numpy()
                train_embeds = torch.cat(self.train_outputs['embeddings_cleancls']).detach().cpu().numpy()
                mean_log.update({'val_fxd_generated_to_allseqs_allTrainSet': get_wasserstein_dist(embeds_gen, train_embeds)})
                if not self.args.target_class == self.model.num_cls:
                    embeds_cls_specific = train_embeds[train_clss == self.args.target_class]
                    mean_log.update({'val_fxd_generated_to_targetclsseqs_allTrainSet': get_wasserstein_dist(embeds_gen, embeds_cls_specific)})
            clss = torch.cat(self.val_outputs['clss_cleancls']).squeeze().detach().cpu().numpy()
            embeds = torch.cat(self.val_outputs['embeddings_cleancls']).detach().cpu().numpy()
            embeds_rand = torch.randint(0,4, size=embeds_gen.shape).numpy()
            mean_log.update({'val_fxd_randseq_to_allseqs': get_wasserstein_dist(embeds_rand, embeds)})
            mean_log.update({'val_fxd_generated_to_allseqs': get_wasserstein_dist(embeds_gen, embeds)})
            if not self.args.target_class == self.model.num_cls:
                embeds_cls_specific = embeds[clss == self.args.target_class]
                mean_log.update({'val_fxd_generated_to_targetclsseqs': get_wasserstein_dist(embeds_gen, embeds_cls_specific)})
            if self.args.taskiran_seq_path is not None:
                embeds_taskiran = torch.cat(self.val_outputs['embeddings_cleancls_taskiran']).detach().cpu().numpy()
                mean_log.update({'val_fxd_taskiran_to_allseqs': get_wasserstein_dist(embeds_taskiran, embeds)})
                if not self.args.target_class == self.model.num_cls:
                    mean_log.update({'val_fxd_taskiran_to_targetclsseqs': get_wasserstein_dist(embeds_taskiran, embeds_cls_specific)})
        self.mean_log_ema = update_ema(current_dict=mean_log, prev_ema=self.mean_log_ema, gamma=0.9)
        mean_log.update(self.mean_log_ema)
        if self.trainer.is_global_zero:
            logger.info(str(mean_log))
            self.log_dict(mean_log, batch_size=1)
            if self.args.wandb:
                wandb.log(mean_log)
                if self.args.dataset_type == 'toy_sampled':
                    pil_dist_comp = self.plot_empirical_and_true(empirical_dist, self.toy_data.probs[self.args.target_class])
                    wandb.log({'fig': [wandb.Image(pil_dist_comp)], 'step': float(self.trainer.global_step), 'iter_step': float(self.iter_step)})
                if self.args.cls_ckpt is not None:
                    pil_probs, pil_score_norms = self.plot_score_and_probs()
                    wandb.log({'fig': [wandb.Image(pil_probs), wandb.Image(pil_score_norms)], 'step': float(self.trainer.global_step), 'iter_step': float(self.iter_step)})

            path = os.path.join(os.environ["MODEL_DIR"], f"val_{self.trainer.global_step}.csv")
            pd.DataFrame(log).to_csv(path)

        for key in list(log.keys()):
            if "val_" in key:
                del self._log[key]
        self.val_outputs = defaultdict(list)



    def on_train_epoch_start(self) -> None:
        self.inf_counter = 1
        self.nan_inf_counter = 0
        if not self.loaded_distill_model and self.args.distill_ckpt is not None:
            self.load_distill_model()
            self.loaded_distill_model = True
        if not self.loaded_classifiers:
            self.load_classifiers(load_cls=self.args.cls_ckpt is not None, load_clean_cls=self.args.clean_cls_ckpt is not None)
            self.loaded_classifiers = True

    def on_train_epoch_end(self):
        self.train_out_initialized = True
        log = self._log
        log = {key: log[key] for key in log if "train_" in key}
        log = self.gather_log(log, self.trainer.world_size)
        mean_log = self.get_log_mean(log)
        mean_log.update(
            {'epoch': float(self.trainer.current_epoch), 'step': float(self.trainer.global_step), 'iter_step': float(self.iter_step)})

        if self.trainer.is_global_zero:
            logger.info(str(mean_log))
            self.log_dict(mean_log, batch_size=1)
            if self.args.wandb:
                wandb.log(mean_log)

        for key in list(log.keys()):
            if "train_" in key:
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

    def plot_empirical_and_true(self, empirical_dist, true_dist):
        num_datasets_to_plot = min(4, empirical_dist.shape[0])
        width = 1
        # Creating a figure and axes
        fig, axes = plt.subplots(math.ceil(num_datasets_to_plot/2), 2, figsize=(10, 8))
        for i in range(num_datasets_to_plot):
            row, col = i // 2, i % 2
            x = np.arange(len(empirical_dist[i]))
            axes[row, col].bar(x, empirical_dist[i], width, label=f'empirical')
            axes[row, col].plot(x, true_dist[i], label=f'true density', color='orange')
            axes[row, col].legend()
            axes[row, col].set_title(f'Sequence position {i + 1}')
            axes[row, col].set_xlabel('Category')
            axes[row, col].set_ylabel('Density')
        plt.tight_layout()
        fig.canvas.draw()
        pil_img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plt.close()
        return pil_img

    def load_model(self, alphabet_size, num_cls):
        if self.args.model == 'cnn':
            self.model = CNNModel(self.args, alphabet_size=alphabet_size, num_cls=num_cls)
        elif self.args.model == 'mlp':
            self.model = MLPModel(self.args, alphabet_size=alphabet_size, num_cls=num_cls)
        elif self.args.model == 'transformer':
            self.model = TransformerModel(self.args, alphabet_size=alphabet_size, num_cls=num_cls)
        elif self.args.model == 'deepflybrain':
            self.model = DeepFlyBrainModel(self.args, alphabet_size=alphabet_size,num_cls=num_cls)
        else:
            raise NotImplementedError()


    def load_classifiers(self, load_cls, load_clean_cls, requires_grad = False):
        if load_cls:
            with open(self.args.cls_ckpt_hparams) as f:
                hparams = yaml.load(f, Loader=yaml.UnsafeLoader)
            if self.args.cls_model == 'cnn':
                self.cls_model = CNNModel(hparams['args'], alphabet_size=self.model.alphabet_size, num_cls=self.model.num_cls, classifier=True)
            elif self.args.cls_model == 'mlp':
                self.cls_model = MLPModel(hparams['args'], alphabet_size=self.model.alphabet_size, num_cls=self.model.num_cls, classifier=True)
            elif self.args.cls_model == 'transformer':
                self.cls_model = TransformerModel(hparams['args'], alphabet_size=self.model.alphabet_size, num_cls=self.model.num_cls, classifier=True)
            elif self.args.cls_model == 'deepflybrain':
                self.cls_model = DeepFlyBrainModel(hparams['args'], alphabet_size=self.model.alphabet_size, num_cls=self.model.num_cls, classifier=True)
            else:
                raise NotImplementedError()
            self.cls_model.load_state_dict(upgrade_state_dict(torch.load(self.args.cls_ckpt, map_location=self.device)['state_dict'],prefixes=['model.']))
            self.cls_model.eval()
            self.cls_model.to(self.device)
            for param in self.cls_model.parameters():
                param.requires_grad = requires_grad

        if  load_clean_cls:
            with open(self.args.clean_cls_ckpt_hparams) as f:
                hparams = yaml.load(f, Loader=yaml.UnsafeLoader)
            if self.args.clean_cls_model == 'cnn':
                self.clean_cls_model = CNNModel(hparams['args'], alphabet_size=self.model.alphabet_size, num_cls=self.model.num_cls, classifier=True)
            elif self.args.clean_cls_model == 'mlp':
                self.clean_cls_model = MLPModel(hparams['args'], alphabet_size=self.model.alphabet_size, num_cls=self.model.num_cls, classifier=True)
            elif self.args.clean_cls_model == 'transformer':
                self.clean_cls_model = TransformerModel(hparams['args'], alphabet_size=self.model.alphabet_size, num_cls=self.model.num_cls, classifier=True)
            elif self.args.clean_cls_model == 'deepflybrain':
                self.clean_cls_model = DeepFlyBrainModel(hparams['args'], alphabet_size=self.model.alphabet_size, num_cls=self.model.num_cls, classifier=True)
            else:
                raise NotImplementedError()
            self.clean_cls_model.load_state_dict(upgrade_state_dict(torch.load(self.args.clean_cls_ckpt, map_location=self.device)['state_dict'],prefixes=['model.']))
            self.clean_cls_model.eval()
            self.clean_cls_model.to(self.device)
            for param in self.clean_cls_model.parameters():
                param.requires_grad = requires_grad

    def load_distill_model(self):
        with open(self.args.distill_ckpt_hparams) as f:
            hparams = yaml.load(f, Loader=yaml.UnsafeLoader)
            self.distill_args = copy.deepcopy(hparams['args'])
        if self.distill_args.model == 'cnn':
            self.distill_model = CNNModel(self.distill_args, alphabet_size=self.model.alphabet_size,num_cls=self.model.num_cls)
        elif self.distill_args.model == 'mlp':
            self.distill_model = MLPModel(self.distill_args, alphabet_size=self.model.alphabet_size,num_cls=self.model.num_cls)
        elif self.distill_args.model == 'transformer':
            self.distill_model = TransformerModel(self.distill_args, alphabet_size=self.model.alphabet_size,num_cls=self.model.num_cls)
        elif self.distill_args.model == 'deepflybrain':
            self.distill_model = DeepFlyBrainModel(self.distill_args, alphabet_size=self.model.alphabet_size,num_cls=self.model.num_cls)
        else:
            raise NotImplementedError()
        upgraded_dict = upgrade_state_dict(torch.load(self.args.distill_ckpt, map_location=self.device)['state_dict'], prefixes=['model.'])
        no_cls_dict = {k: v for k, v in upgraded_dict.items() if 'cls_model' not in k}
        self.distill_model.load_state_dict(no_cls_dict)
        self.distill_model.eval()
        self.distill_model.to(self.device)
        for param in self.distill_model.parameters():
            param.requires_grad = False

    def plot_score_and_probs(self):
        clss = torch.cat(self.val_outputs['clss_noisycls'])
        probs = torch.softmax(torch.cat(self.val_outputs['logits_noisycls']), dim=-1)
        scores = torch.cat(self.val_outputs['scores_noisycls']).cpu().numpy()
        score_norms = np.linalg.norm(scores, axis=-1)
        alphas = torch.cat(self.val_outputs['alphas_noisycls']).cpu().numpy()
        true_probs = probs[torch.arange(len(probs)), clss].cpu().numpy()
        bins = np.linspace(min(alphas), 12, 20)
        indices = np.digitize(alphas, bins)
        bin_means = [np.mean(true_probs[indices == i]) for i in range(1, len(bins))]
        bin_std = [np.std(true_probs[indices == i]) for i in range(1, len(bins))]
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        bin_pos_std = [np.std(true_probs[indices == i][true_probs[indices == i] > np.mean(true_probs[indices == i])]) for i in range(1, len(bins))]
        bin_neg_std = [np.std(true_probs[indices == i][true_probs[indices == i] < np.mean(true_probs[indices == i])]) for i in range(1, len(bins))]
        plot_data = pd.DataFrame({'Alphas': bin_centers, 'Means': bin_means, 'Std': bin_std, 'Pos_Std': bin_pos_std, 'Neg_Std': bin_neg_std})
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='Alphas', y='Means', data=plot_data)
        plt.fill_between(plot_data['Alphas'], plot_data['Means'] - plot_data['Neg_Std'], plot_data['Means'] + plot_data['Pos_Std'], alpha=0.3)
        plt.xlabel('Binned alphas values')
        plt.ylabel('Mean of predicted probs for true class')
        fig = plt.gcf()
        fig.canvas.draw()
        pil_probs = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

        plt.close()
        bin_means = [np.mean(score_norms[indices == i]) for i in range(1, len(bins))]
        bin_std = [np.std(score_norms[indices == i]) for i in range(1, len(bins))]
        bin_pos_std = [np.std(score_norms[indices == i][score_norms[indices == i] > np.mean(score_norms[indices == i])]) for i in range(1, len(bins))]
        bin_neg_std = [np.std(score_norms[indices == i][score_norms[indices == i] < np.mean(score_norms[indices == i])]) for i in range(1, len(bins))]
        plot_data = pd.DataFrame({'Alphas': bin_centers, 'Means': bin_means, 'Std': bin_std, 'Pos_Std': bin_pos_std, 'Neg_Std': bin_neg_std})
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='Alphas', y='Means', data=plot_data)
        plt.fill_between(plot_data['Alphas'], plot_data['Means'] - plot_data['Neg_Std'],
                         plot_data['Means'] + plot_data['Pos_Std'], alpha=0.3)
        plt.xlabel('Binned alphas values')
        plt.ylabel('Mean of norm of the scores')
        fig = plt.gcf()
        fig.canvas.draw()
        pil_score_norms = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        return pil_probs, pil_score_norms

    def log_data_similarities(self, seq_pred):
        similarities1 = seq_pred.cpu()[:, None, :].eq(self.toy_data.data_class1[None, :, :])  # batchsize, dataset_size, seq_len
        similarities2 = seq_pred.cpu()[:, None, :].eq(self.toy_data.data_class2[None, :, :])  # batchsize, dataset_size, seq_len
        similarities = seq_pred.cpu()[:, None, :].eq(torch.cat([self.toy_data.data_class2[None, :, :], self.toy_data.data_class1[None, :, :]],dim=1))  # batchsize, dataset_size, seq_len
        self.lg('data1_sim', similarities1.float().mean(-1).max(-1)[0])
        self.lg('data2_sim', similarities2.float().mean(-1).max(-1)[0])
        self.lg('data_sim', similarities.float().mean(-1).max(-1)[0])
        self.lg('mean_data1_sim', similarities1.float().mean(-1).mean(-1))
        self.lg('mean_data2_sim', similarities2.float().mean(-1).mean(-1))
        self.lg('mean_data_sim', similarities.float().mean(-1).mean(-1))
