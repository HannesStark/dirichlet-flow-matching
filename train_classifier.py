from lightning_modules.cls_module import CLSModule
from utils.dataset import ToyDataset, TwoClassOverfitDataset, EnhancerDataset
from utils.parsing import parse_train_args

args = parse_train_args()

import torch, os, wandb
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
            monitor= 'val_loss' if args.val_loss_es else'val_accuracy',
            mode = 'min' if args.val_loss_es else "max"
        )

    ],
    val_check_interval=args.val_check_interval,
    check_val_every_n_epoch=args.check_val_every_n_epoch,
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

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=args.dataset_type == 'enhancer')
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)
model = CLSModule(args, alphabet_size=train_ds.alphabet_size, num_cls=train_ds.num_cls)

if args.validate:
    trainer.validate(model, train_loader if args.validate_on_train else val_loader, ckpt_path=args.ckpt)
else:
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt)
