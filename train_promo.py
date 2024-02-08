from utils.parsing import parse_train_args
from lightning_modules.promoter_module import PromoterModule
from utils.promoter_dataset import PromoterDataset

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
            save_top_k=1,
            save_last=True,
            monitor='val_sp-mse'
        )
    ],
    check_val_every_n_epoch=args.check_val_every_n_epoch,
)


train_ds = PromoterDataset(n_tsses=100000, rand_offset=10, split='train')
val_ds = PromoterDataset(n_tsses=100000, rand_offset=0, split='valid' if not args.validate_on_test else 'test') if not args.overfit else train_ds

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=not args.overfit, num_workers=args.num_workers)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
print("Len train_ds: ", len(train_ds))
print("Len val_ds: ", len(val_ds))
model = PromoterModule(args)

if args.validate:
    trainer.validate(model, train_loader if args.validate_on_train else val_loader, ckpt_path=args.ckpt)
else:
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt)
