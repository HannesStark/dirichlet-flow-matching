import sys
from argparse import ArgumentParser
import subprocess, os
from datetime import datetime




def parse_train_args():
    parser = ArgumentParser()
    
    # Run settings
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--cls_ckpt", type=str, default=None)
    parser.add_argument("--cls_ckpt_hparams", type=str, default=None)
    parser.add_argument("--clean_cls_ckpt", type=str, default=None, help='cls model for evaluation purposes')
    parser.add_argument("--clean_cls_ckpt_hparams", type=str, default=None)
    parser.add_argument("--distill_ckpt", type=str, default=None, help='cls model for evaluation purposes')
    parser.add_argument("--distill_ckpt_hparams", type=str, default=None)
    parser.add_argument("--ckpt_has_cls", action='store_true')
    parser.add_argument("--ckpt_has_clean_cls", action='store_true')
    parser.add_argument("--validate", action='store_true')
    parser.add_argument("--subset_train_as_val", action='store_true')
    parser.add_argument("--validate_on_train", action='store_true')
    parser.add_argument("--validate_on_test", action='store_true')

    # Training
    parser.add_argument("--limit_train_batches", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--constant_val_len", type=int, default=None)
    parser.add_argument("--accumulate_grad", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.)
    parser.add_argument("--lr_multiplier", type=float, default=1.0)
    parser.add_argument("--check_grad", action="store_true")
    parser.add_argument("--no_lr_scheduler", action="store_true")
    parser.add_argument("--checkpoint_layers", action="store_true")
    parser.add_argument("--max_steps", type=int, default=450000)
    parser.add_argument("--max_epochs", type=int, default=100000)

    # Promoter Design Training
    parser.add_argument("--lr", type=float, default=5e-4)

    # Validate
    parser.add_argument("--check_val_every_n_epoch", type=int, default=None)
    parser.add_argument("--limit_val_batches", type=int, default=None)
    parser.add_argument("--fid_early_stop", action="store_true")
    parser.add_argument("--val_loss_es", action="store_true", help='only for cls train')
    parser.add_argument("--val_check_interval", type=int, default=None)
    parser.add_argument("--ckpt_iterations", type=int, nargs='+', default=None)
    parser.add_argument("--random_sequences", action="store_true")
    parser.add_argument("--taskiran_seq_path", type=str, default=None)

    # Data
    parser.add_argument('--dataset_type', type=str, choices=['enhancer', 'toy_fixed','toy_sampled'], default='argmax')
    parser.add_argument("--mel_enhancer", action='store_true')
    parser.add_argument("--overfit", action='store_true')
    parser.add_argument("--promoter_dataset", action='store_true')
    parser.add_argument("--toy_simplex_dim", type=int, default=4)
    parser.add_argument("--toy_num_cls", type=int, default=3)
    parser.add_argument("--toy_num_seq", type=int, default=1000)
    parser.add_argument("--toy_seq_len", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)

    # Guidance
    parser.add_argument("--cls_guidance", action='store_true')
    parser.add_argument("--binary_guidance", action='store_true', help='the model is trained with only the target class and the auxiliary class')
    parser.add_argument("--oversample_target_class", action='store_true')
    parser.add_argument("--target_class", type=int, default=0)
    parser.add_argument("--all_class_inference", action='store_true', help='ignores target_class and guides towards all classes during inference. Helfpul for seeing if we improve the general FID.')
    parser.add_argument("--cls_free_noclass_ratio", type=float, default=0.3)
    parser.add_argument("--cls_free_guidance", action='store_true')
    parser.add_argument("--probability_addition", action='store_true', help='if this is activated then cls_free_guidance also needs to be activated and we then do it with probs tilting instead of with score conversion')
    parser.add_argument("--adaptive_prob_add", action='store_true', help='if this is activated then cls_free_guidance also needs to be activated and we then do it with probs tilting instead of with score conversion')
    parser.add_argument("--vectorfield_addition", action='store_true', help='if this is activated then cls_free_guidance also needs to be activated and we then do it with probs tilting instead of with score conversion')
    parser.add_argument("--probability_tilt", action='store_true', help='if this is activated then cls_free_guidance also needs to be activated and we then do it with probs tilting instead of with score conversion')
    parser.add_argument("--score_free_guidance", action='store_true')
    parser.add_argument("--guidance_scale", type=float, default=0.5)
    parser.add_argument("--analytic_cls_score", action='store_true', help='this only works with the two_class_dataset')
    parser.add_argument("--scale_cls_score", action='store_true')
    parser.add_argument("--allow_nan_cfactor", action="store_true")

    # Model
    parser.add_argument("--model", choices=['8M', '35M', '150M', '650M', '3B', '15B','mlp','cnn','transformer', 'deepflybrain'], default='650M')
    parser.add_argument("--cls_model", choices=['mlp','cnn','transformer', 'deepflybrain'], default='cnn')
    parser.add_argument("--clean_cls_model", choices=['mlp', 'cnn', 'transformer', 'deepflybrain'], default='cnn')
    parser.add_argument("--clean_data", action="store_true", help='do not noise to the model input. E.g. for training a clean calssifier.')
    parser.add_argument("--mode", choices=['distill', 'dirichlet', 'riemannian', 'mlm', 'ardm', 'lrar', 'cdcd', 'ppl_eval'], default='dirichlet')
    parser.add_argument("--simplex_spacing", type=int, default=1000)
    parser.add_argument("--prior_pseudocount", type=float, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_cnn_stacks", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--hidden_dim", type=float, default=128)
    parser.add_argument("--self_condition_ratio", type=float, default=0)
    parser.add_argument("--prior_self_condition", action="store_true")
    parser.add_argument("--no_token_dropout", action="store_true")
    parser.add_argument("--time_embed", action="store_true")
    parser.add_argument("--fix_alpha", type=float, default=None)
    parser.add_argument("--alpha_scale", type=float, default=2)
    parser.add_argument("--alpha_max", type=float, default=8)
    parser.add_argument("--cls_expanded_simplex", action="store_true")
    parser.add_argument("--simplex_encoding_dim", type=int, default=64)
    parser.add_argument("--flow_temp", type=float, default=1.0)
    parser.add_argument('--val_pred_type', type=str, choices=['argmax', 'sample'], default='argmax')
    parser.add_argument('--num_integration_steps', type=int, default=20, help='The number of integration steps used during inference.')

    # Logging
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default="default")
    
    args = parser.parse_args()
    timestamp = datetime.fromtimestamp(datetime.now().timestamp()).strftime("%Y-%m-%d_%H-%M-%S")
    os.environ["MODEL_DIR"] = os.path.join("workdir", (args.run_name + '_' + timestamp))
    os.environ["WANDB_LOGGING"] = str(int(args.wandb))

    from utils.logging import Logger
    os.makedirs(os.environ['MODEL_DIR'], exist_ok=True)
    sys.stdout = Logger(logpath=os.path.join(os.environ['MODEL_DIR'], f'log.log'), syspart=sys.stdout)
    sys.stdout.encoding = None # for pytorch lightning because it is stupid
    sys.stderr = Logger(logpath=os.path.join(os.environ['MODEL_DIR'], f'log.log'), syspart=sys.stderr)
    if args.wandb:
        if subprocess.check_output(["git", "status", "-s"]):
            print('There were uncommited changes. Not running that stuff.')
            exit()
    args.commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    )
    if args.score_free_guidance:
        assert args.cls_free_noclass_ratio == 0, 'no auxiliary class is needed for classifier free guidance if you do score free guidance. The training on the auxialiary classis is basically only data augmentation then.'
    if args.probability_tilt:
        assert args.cls_free_guidance
    assert not (args.probability_tilt and args.score_free_guidance)
    return args