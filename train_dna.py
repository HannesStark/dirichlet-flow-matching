from torch.utils.data import WeightedRandomSampler, Subset

from lightning_modules.dna_module import DNAModule
from utils.dataset import ToyDataset, TwoClassOverfitDataset, EnhancerDataset
from utils.parsing import parse_train_args
args = parse_train_args()
import torch, os, wandb
torch.manual_seed(0)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

if args.wandb:
    wandb.init(
        entity="anonymized",
        settings=wandb.Settings(start_method="fork"),
        project="betawolf",
        name=args.run_name,
        config=args,
    )

trainer = pl.Trainer(
    default_root_dir=os.environ["MODEL_DIR"],
    accelerator="gpu" if torch.cuda.is_available() else 'auto',
    max_steps=args.max_steps,
    max_epochs=args.max_epochs,
    num_sanity_val_steps=0,
    limit_train_batches=args.limit_train_batches,
    limit_val_batches=args.limit_val_batches,
    enable_progress_bar=not (args.wandb or args.no_tqdm) or os.getlogin() == 'anonymized',
    gradient_clip_val=args.grad_clip,
    callbacks=[
        ModelCheckpoint(
            dirpath=os.environ["MODEL_DIR"],
            save_top_k=5,
            save_last=True,
            monitor='val_fxd_generated_to_allseqs' if args.fid_early_stop else 'val_perplexity',
            mode = "min"
        )
    ],
    check_val_every_n_epoch=args.check_val_every_n_epoch,
    val_check_interval=args.val_check_interval,
)


if args.dataset_type == 'toy_fixed':
    train_ds = TwoClassOverfitDataset(args)
    val_ds = train_ds
    toy_data = train_ds
elif args.dataset_type == 'toy_sampled':
    train_ds = ToyDataset(args)
    val_ds = train_ds
    toy_data = train_ds
elif args.dataset_type == 'enhancer':
    train_ds = EnhancerDataset(args, split='train')
    val_ds = EnhancerDataset(args, split='valid' if not args.validate_on_test else 'test')
    toy_data = None

if args.subset_train_as_val:
    val_set_size = len(val_ds) if args.constant_val_len is None else args.constant_val_len
    val_ds = Subset(train_ds, torch.randperm(len(train_ds))[:val_set_size])

if args.oversample_target_class:
    weights = torch.zeros(len(train_ds))
    is_target_cls = train_ds.clss == args.target_class
    weights[is_target_cls] = 0.5 / is_target_cls.sum()
    weights[~is_target_cls] = 0.5 / (~is_target_cls).sum()
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_ds), replacement=True)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, sampler=sampler)
else:
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=args.dataset_type == 'enhancer')
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)

model = DNAModule(args, train_ds.alphabet_size, train_ds.num_cls, toy_data)

if args.validate:
    trainer.validate(model, train_loader if args.validate_on_train else val_loader, ckpt_path=args.ckpt)
else:
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt)
