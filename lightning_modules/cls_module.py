from collections import defaultdict

import PIL
import numpy as np
import pandas as pd
import torch, time, os
import torchmetrics
import wandb
from matplotlib import pyplot as plt
from torch import optim
import seaborn as sns

from utils.visualize import create_scatter_plot

sns.set_style('whitegrid')
from model.dna_models import MLPModel, CNNModel, TransformerModel, DeepFlyBrainModel
from utils.flow_utils import DirichletConditionalFlow, expand_simplex, sample_cond_prob_path
from lightning_modules.general_module import GeneralModule
from utils.logging import get_logger


logger = get_logger(__name__)


class CLSModule(GeneralModule):
    def __init__(self, args, alphabet_size, num_cls):
        super().__init__(args)
        if self.args.cls_model == 'cnn':
            self.model = CNNModel(args, alphabet_size=alphabet_size,num_cls=num_cls, classifier=True)
        elif self.args.cls_model == 'mlp':
            self.model = MLPModel(args, alphabet_size=alphabet_size,num_cls=num_cls, classifier=True)
        elif self.args.cls_model == 'transformer':
            self.model = TransformerModel(args, alphabet_size=alphabet_size,num_cls=num_cls, classifier=True)
        elif self.args.cls_model == 'deepflybrain':
            self.model = DeepFlyBrainModel(args, alphabet_size=alphabet_size,num_cls=num_cls, classifier=True)
        else:
            raise NotImplementedError()
        self.condflow = DirichletConditionalFlow(K=self.model.alphabet_size, alpha_spacing=0.01, alpha_max=args.alpha_max)
        self.crossent_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.val_output = defaultdict(list)

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
        cls = cls.squeeze()
        B, L = seq.shape

        xt, alphas = sample_cond_prob_path(self.args, seq, self.model.alphabet_size)
        xt_inp = xt
        if self.args.cls_expanded_simplex:
            xt_inp, prior_weights = expand_simplex(xt, alphas, self.args.prior_pseudocount)
        logits = self.model(xt_inp if not self.args.clean_data else seq, t=alphas)

        losses = self.crossent_loss(logits, cls)

        self.lg('loss', losses)
        probs = torch.softmax(logits,dim=-1)
        if self.args.val_pred_type == 'argmax':
            cls_pred = torch.argmax(logits, dim=-1)
        elif self.args.val_pred_type == 'sample':
            cls_pred = torch.nn.functional.softmax(logits, dim=-1)
            cls_pred = torch.distributions.Categorical(cls_pred).sample()
        if self.stage == 'val':
            self.val_output['clss'].append(cls)
            self.val_output['logits'].append(logits)
            self.val_output['alphas'].append(alphas)
            if not self.args.clean_data:
                scores = self.get_cls_score(xt, alphas)
                self.val_output['scores'].append(scores)
        self.lg('accuracy', cls_pred.eq(cls).float())
        self.lg('alpha', alphas)
        if self.args.cls_expanded_simplex: self.lg('prior_weight', prior_weights)
        self.lg('dur', torch.tensor(time.time() - self.last_log_time)[None].expand(B))
        self.last_log_time = time.time()
        return losses.mean()

    def get_cls_score(self, xt, alpha):
        with torch.enable_grad():
            xt_ = xt.clone().detach().requires_grad_(True)
            if self.args.cls_expanded_simplex:
                prior_weight = self.args.prior_pseudocount / (alpha[:,None,None] + self.args.prior_pseudocount - 1)
                xt_ = torch.cat([xt_ * (1 - prior_weight), xt_ * prior_weight], -1)

            cls_logits = self.model(xt_, t=alpha)
            loss = self.crossent_loss(cls_logits, torch.zeros(len(xt), dtype=torch.long, device=xt.device)).mean()
            assert not torch.isnan(loss).any()
            cls_score = torch.autograd.grad(loss,[xt_])[0]
            assert not torch.isnan(cls_score).any()
        cls_score = cls_score - cls_score.mean(-1)[:,:,None]
        return cls_score.detach()

    def on_validation_epoch_end(self):
        self.generator = np.random.default_rng()
        log = self._log
        log = {key: log[key] for key in log if "val_" in key}
        log = self.gather_log(log, self.trainer.world_size)
        mean_log = self.get_log_mean(log)
        mean_log.update({'epoch': float(self.trainer.current_epoch), 'step': float(self.trainer.global_step), 'iter_step': float(self.iter_step)})

        pil_auroc_aupr, pil_auroc_acc, pil_acc_aupr, aurocs, accuracies, auprs = self.scatter_plots()
        mean_log.update({'max_auroc': float(aurocs.max()), 'max_aupr': float(auprs.max()), 'max_accuracy': float(accuracies.max())})

        if self.trainer.is_global_zero:
            logger.info(str(mean_log))
            self.log_dict(mean_log, batch_size=1)
            if self.args.wandb:

                wandb.log({'fig': [wandb.Image(pil_auroc_aupr), wandb.Image(pil_auroc_acc), wandb.Image(pil_acc_aupr)], 'step': self.trainer.global_step,'iter_step': self.iter_step})
                if not self.args.clean_data:
                    pil_img, pil_img2 = self.plot_probs_per_alpha()
                    wandb.log({'fig': [wandb.Image(pil_img), wandb.Image(pil_img2)], 'step': self.trainer.global_step, 'iter_step': self.iter_step})
                wandb.log(mean_log)
            pd.DataFrame(log).to_csv(os.path.join(os.environ["MODEL_DIR"], f"val_{self.trainer.global_step}.csv"))

        for key in list(log.keys()):
            if "val_" in key:
                del self._log[key]
        self.val_output = defaultdict(list)

    def scatter_plots(self):
        clss = torch.cat(self.val_output['clss'])
        probs = torch.softmax(torch.cat(self.val_output['logits']), dim=-1)
        aurocs = torchmetrics.classification.MulticlassAUROC(num_classes=self.model.num_cls, average=None).to(self.device)(probs, clss).detach().cpu().numpy()
        accuracies = torchmetrics.classification.MulticlassAccuracy(num_classes=self.model.num_cls, average=None).to(self.device)(probs, clss).detach().cpu().numpy()
        auprs = torchmetrics.classification.MulticlassAveragePrecision(num_classes=self.model.num_cls, average=None).to(self.device)(probs,clss).detach().cpu().numpy()
        title = f'{"Melanoma cells" if self.args.mel_enhancer else "Fly Brain cells"}'
        pil_auroc_aupr = create_scatter_plot(x=aurocs, y=auprs, title=title, x_label='auROC', y_label='auPR')
        pil_auroc_acc = create_scatter_plot(x=aurocs, y=accuracies, title=title, x_label='auROC', y_label='accuracy')
        pil_acc_aupr = create_scatter_plot(x=accuracies, y=auprs, title=title, x_label='accuracy', y_label='auPR')
        return pil_auroc_aupr, pil_auroc_acc, pil_acc_aupr, aurocs, accuracies, auprs


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

    def plot_probs_per_alpha(self):
        clss = torch.cat(self.val_output['clss'])
        probs = torch.softmax(torch.cat(self.val_output['logits']), dim=-1)
        scores = torch.cat(self.val_output['scores']).cpu().numpy()
        score_norms = np.linalg.norm(scores, axis=-1)
        alphas = torch.cat(self.val_output['alphas']).cpu().numpy()
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
        pil_img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

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
        pil_img2 = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())


        return pil_img, pil_img2
