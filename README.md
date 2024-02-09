# Dirichlet Flow Matching with Applications to DNA Sequence Design

### [Paper on arXiv](http://arxiv.org/abs/2402.05841)

### Conda environment
```yaml
conda create -c conda-forge -n seq python=3.9
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
pip install torch_geometric jupyterlab gpustat pyyaml wandb biopython spyrmsd einops biopandas plotly seaborn prody tqdm lightning imageio tmtools "fair-esm[esmfold]" e3nn
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.1.0+cu113.htm

# The libraries below are required for the promoter design experiments
git clone https://github.com/kathyxchen/selene.git
cd selene
git checkout custom_target_support
python setup.py build_ext --inplace
python setup.py install

pip install pyBigWig pytabix cooler pyranges biopython cooltools
```

# Experiments
We provide the weights of trained models for all experiments (if required). Unzip them into in `workdir`.
```
https://publbuck.s3.us-east-2.amazonaws.com/workdir.zip
```

## Toy experiments
The commands below are for linear flow matching (mode riemannian) and dirichlet flow matching. K in the paper corresponds to `--toy_simplex_dim` here.
```yaml
python -m train_dna --run_name trainToy_linear_dim40 --dataset_type toy_sampled --limit_val_batches 1000 --toy_seq_len 4 --toy_simplex_dim 40 --toy_num_cls 1 --val_check_interval 5000 --batch_size 512 --print_freq 100 --wandb --model cnn --mode riemannian

python -m train_dna --run_name trainToy_diri_dim40 --dataset_type toy_sampled --limit_val_batches 1000 --toy_seq_len 4 --toy_simplex_dim 40 --toy_num_cls 1 --val_check_interval 5000 --batch_size 512 --print_freq 100 --wandb --model cnn

```

## Promoter design experiments (Table 1)

Download the dataset from https://zenodo.org/records/7943307 and place it in `data`.

Example command for retraining:

```yaml
python -m train_promo --run_name train_dirichlet_fm --batch_size 128 --wandb --num_workers 4 --check_val_every_n_epoch 5 --num_integration_steps 100 --limit_val_batches 16
```

Commands for running inference with the provided model weights:
Here `lrar` corresponds to the language model and `riemannian` to linear flow matching:

```yaml
python -m train_promo --run_name dirichlet_flow_matching_distilled --batch_size 128 --wandb --num_workers 4 --num_integration_steps 100 --ckpt workdir/promo_distill_diri_2024-01-09_16-53-39/epoch=14-step=10380.ckpt --validate --validate_on_test --mode distill

python -m train_promo --run_name dirichlet_flow_matching --batch_size 128 --wandb --num_workers 4 --check_val_every_n_epoch 5 --num_integration_steps 100 --validate --validate_on_test --ckpt workdir/promo_diri_2024-01-31_10-49-42/epoch=14-step=10380-Copy1.ckpt

python -m train_promo --run_name language_model --batch_size 128 --wandb --num_workers 4 --check_val_every_n_epoch 5 --num_integration_steps 100 --mode lrar --validate --validate_on_test --ckpt workdir/promo_lrar_sani_2024-01-31_10-46-33/epoch=69-step=24220-Copy1.ckpt

python -m train_promo --run_name linear_flow_matching --batch_size 128 --wandb --num_workers 4 --check_val_every_n_epoch 5 --num_integration_steps 100 --mode riemannian --validate --validate_on_test --ckpt workdir/promo_riem_sani_2024-01-31_10-55-43/epoch=124-step=43250-Copy1.ckpt
```


## Enhancer design Experiments

Download the dataset from https://zenodo.org/records/10184648 and place it into `data` to have the path `data/the_code/...`


The following is an example command for training to then carry out the classifier free guidance experiments:

```yaml
python -m train_dna --run_name train_FB_dirichlet_fm_cfguidance3 --batch_size 256 --print_freq 200 --wandb --dataset_type enhancer --num_integration_steps 100 --model cnn --num_cnn_stacks 4 --cls_free_guidance --guidance_scale 3 --clean_cls_ckpt_hparams workdir/clsDNAclean_cnn_1stack_2023-12-30_15-01-30/lightning_logs/version_0/hparams.yaml --clean_cls_ckpt workdir/clsDNAclean_cnn_1stack_2023-12-30_15-01-30/epoch=15-step=10480.ckpt --target_class 2 --check_val_every_n_epoch 10 --subset_train_as_val --allow_nan_cfactor --all_class_inference --probability_addition --fid_early_stop
```

Here are the commands for Table 2. MEL corresponds to the melanoma data and FB to the flybrain data

```yaml
python -m train_dna --run_name FB_dirichlet_fm_cfguidance3 --batch_size 256 --print_freq 200 --wandb --dataset_type enhancer --num_integration_steps 100 --model cnn --num_cnn_stacks 4 --cls_free_guidance --guidance_scale 3 --clean_cls_ckpt_hparams workdir/clsDNAclean_cnn_1stack_2023-12-30_15-01-30/lightning_logs/version_0/hparams.yaml --clean_cls_ckpt workdir/clsDNAclean_cnn_1stack_2023-12-30_15-01-30/epoch=15-step=10480.ckpt --target_class 2 --check_val_every_n_epoch 10 --subset_train_as_val --allow_nan_cfactor --all_class_inference --probability_addition --fid_early_stop --validate --validate_on_test --ckpt workdir/DNA_valOnTrain_epoch12Eval_uncondFID_FIDearlyS_probAddGuidance3_2024-01-07_16-23-01/epoch=1329-step=436240.ckpt

python -m train_dna --run_name FB_linear_fm --batch_size 256 --print_freq 200 --wandb --dataset_type enhancer --num_integration_steps 100 --model cnn --num_cnn_stacks 4 --clean_cls_ckpt_hparams workdir/clsDNAclean_cnn_1stack_2023-12-30_15-01-30/lightning_logs/version_0/hparams.yaml --clean_cls_ckpt workdir/clsDNAclean_cnn_1stack_2023-12-30_15-01-30/epoch=15-step=10480.ckpt --target_class 2 --check_val_every_n_epoch 10 --subset_train_as_val --mode riemannian --fid_early_stop --max_epochs 800 --validate --validate_on_test --ckpt workdir/DNA_riem_target2_valOnTrain_noDropoutEval_2024-01-08_10-11-24/epoch=479-step=157440.ckpt

python -m train_dna --run_name FB_dirichlet_fm --batch_size 256 --print_freq 200 --wandb --dataset_type enhancer --num_integration_steps 100 --model cnn --num_cnn_stacks 4 --cls_free_guidance --guidance_scale 0 --clean_cls_ckpt_hparams workdir/clsDNAclean_cnn_1stack_2023-12-30_15-01-30/lightning_logs/version_0/hparams.yaml --clean_cls_ckpt workdir/clsDNAclean_cnn_1stack_2023-12-30_15-01-30/epoch=15-step=10480.ckpt --target_class 2 --check_val_every_n_epoch 10 --subset_train_as_val --allow_nan_cfactor --all_class_inference --probability_addition --fid_early_stop --validate --validate_on_test --ckpt workdir/DNA_valOnTrain_epoch12Eval_uncondFID_FIDearlyS_probAddGuidance3_2024-01-07_16-23-01/epoch=1329-step=436240.ckpt

python -m train_dna --run_name FB_language_model --batch_size 256 --print_freq 200 --wandb --dataset_type enhancer --num_integration_steps 100 --model cnn --num_cnn_stacks 4 --clean_cls_ckpt_hparams workdir/clsDNAclean_cnn_1stack_2023-12-30_15-01-30/lightning_logs/version_0/hparams.yaml --clean_cls_ckpt workdir/clsDNAclean_cnn_1stack_2023-12-30_15-01-30/epoch=15-step=10480.ckpt --target_class 2 --check_val_every_n_epoch 10 --subset_train_as_val --mode lrar --fid_early_stop --max_epochs 800 --validate --validate_on_test --ckpt workdir/DNA_lrar_target2_valOnTrain_noDropoutEval_2024-01-08_10-11-21/epoch=49-step=16400.ckpt

python -m train_dna --run_name FB_dirichlet_fm_distilled --batch_size 256 --print_freq 200 --wandb --dataset_type enhancer --num_integration_steps 100 --model cnn --num_cnn_stacks 4 --guidance_scale 3 --clean_cls_ckpt_hparams workdir/clsDNAclean_cnn_1stack_2023-12-30_15-01-30/lightning_logs/version_0/hparams.yaml --clean_cls_ckpt workdir/clsDNAclean_cnn_1stack_2023-12-30_15-01-30/epoch=15-step=10480.ckpt --target_class 2 --check_val_every_n_epoch 1 --subset_train_as_val --allow_nan_cfactor --all_class_inference --probability_addition --fid_early_stop --mode distill --distill_ckpt workdir/DNA_diri_target2_valOnTrain_epoch12Eval_2024-01-07_14-27-29/epoch=409-step=134480.ckpt --distill_ckpt_hparams workdir/DNA_diri_target2_valOnTrain_epoch12Eval_2024-01-07_14-27-29/lightning_logs/version_0/hparams.yaml --ckpt workdir/DNA_valOnTrain_epoch12Eval_DISTILL_2024-01-15_15-38-23/epoch=334-step=109880-Copy1.ckpt --validate --validate_on_test


python -m train_dna --run_name MEL_language_model --batch_size 256 --print_freq 200 --wandb --dataset_type enhancer --num_integration_steps 100 --model cnn --num_cnn_stacks 4 --clean_cls_ckpt_hparams workdir/clsMELclean_cnn_dropout02_2023-12-31_12-26-28/lightning_logs/version_0/hparams.yaml --clean_cls_ckpt workdir/clsMELclean_cnn_dropout02_2023-12-31_12-26-28/epoch=9-step=5540.ckpt --target_class 13 --check_val_every_n_epoch 10 --mel_enhancer --subset_train_as_val --mode lrar --fid_early_stop --max_epochs 800 --validate_on_test --validate --ckpt workdir/MEL_lrar_target13_valOnTrain_earlyStopEval_2024-01-08_10-11-16/epoch=29-step=8310.ckpt

python -m train_dna --run_name MEL_linear_fm --batch_size 256 --print_freq 200 --wandb --dataset_type enhancer --num_integration_steps 100 --model cnn --num_cnn_stacks 4 --clean_cls_ckpt_hparams workdir/clsMELclean_cnn_dropout02_2023-12-31_12-26-28/lightning_logs/version_0/hparams.yaml --clean_cls_ckpt workdir/clsMELclean_cnn_dropout02_2023-12-31_12-26-28/epoch=9-step=5540.ckpt --target_class 13 --check_val_every_n_epoch 10 --mel_enhancer --subset_train_as_val --mode riemannian --fid_early_stop --max_epochs 800 --validate_on_test --validate --ckpt workdir/MEL_riem_target13_valOnTrain_earlyStopEval_2024-01-08_10-11-17/epoch=59-step=16620.ckpt

python -m train_dna --run_name MEL_dirichlet_fm --batch_size 256 --print_freq 200 --wandb --dataset_type enhancer --num_integration_steps 100 --model cnn --num_cnn_stacks 4 --cls_free_guidance --guidance_scale 0 --clean_cls_ckpt_hparams workdir/clsMELclean_cnn_dropout02_2023-12-31_12-26-28/lightning_logs/version_0/hparams.yaml --clean_cls_ckpt workdir/clsMELclean_cnn_dropout02_2023-12-31_12-26-28/epoch=9-step=5540.ckpt --target_class 13 --check_val_every_n_epoch 10 --subset_train_as_val --allow_nan_cfactor --all_class_inference --probability_addition --fid_early_stop --mel_enhancer --ckpt workdir/MEL_valOnTrain_epoch12Eval_uncondFID_FIDearlyS_probAddGuidance1_2024-01-08_16-29-33/epoch=1399-step=387800.ckpt --validate --validate_on_test

python -m train_dna --run_name MEL_dirichlet_fm_cfguidance3 --batch_size 256 --print_freq 200 --wandb --dataset_type enhancer --num_integration_steps 100 --model cnn --num_cnn_stacks 4 --cls_free_guidance --guidance_scale 2 --clean_cls_ckpt_hparams workdir/clsMELclean_cnn_dropout02_2023-12-31_12-26-28/lightning_logs/version_0/hparams.yaml --clean_cls_ckpt workdir/clsMELclean_cnn_dropout02_2023-12-31_12-26-28/epoch=9-step=5540.ckpt --target_class 13 --check_val_every_n_epoch 10 --subset_train_as_val --allow_nan_cfactor --all_class_inference --probability_addition --fid_early_stop --mel_enhancer --ckpt workdir/MEL_valOnTrain_epoch12Eval_uncondFID_FIDearlyS_probAddGuidance1_2024-01-08_16-29-33/epoch=1399-step=387800.ckpt --validate --validate_on_test

python -m train_dna --run_name MEL_dirichlet_fm_distilled --batch_size 256 --print_freq 200 --wandb --dataset_type enhancer --num_integration_steps 100 --model cnn --num_cnn_stacks 4 --clean_cls_ckpt_hparams workdir/clsMELclean_cnn_dropout02_2023-12-31_12-26-28/lightning_logs/version_0/hparams.yaml --clean_cls_ckpt workdir/clsMELclean_cnn_dropout02_2023-12-31_12-26-28/epoch=9-step=5540.ckpt --target_class 13 --check_val_every_n_epoch 1 --mel_enhancer --subset_train_as_val --mode dirichlet --fid_early_stop --max_epochs 800 --mode distill --distill_ckpt workdir/MEL_diri_target13_valOnTrain_earlyStopEval_2024-01-08_12-47-07/epoch=729-step=202210.ckpt --distill_ckpt_hparams workdir/MEL_diri_target13_valOnTrain_earlyStopEval_2024-01-08_12-47-07/lightning_logs/version_0/hparams.yaml --ckpt workdir/resume_resume_MEL_diri_target13_valOnTrain_earlyStopEval_DISTILL_2024-01-25_13-41-13/epoch=194-step=54015.ckpt --validate --validate_on_test


```

Here are the commands for class conditioned classifier free guidance experiments (vary the `--guidance_scale` for changing gamma and `--target_class` for the different target classes). The classes we used are (35, 2, 68, 16):

```yaml
python -m train_dna --run_name dirichlet_fm_no_guidance --batch_size 256 --print_freq 200 --wandb --dataset_type enhancer --num_integration_steps 100 --model cnn --num_cnn_stacks 4 --clean_cls_ckpt_hparams workdir/clsDNAclean_cnn_1stack_2023-12-30_15-01-30/lightning_logs/version_0/hparams.yaml --clean_cls_ckpt workdir/clsDNAclean_cnn_1stack_2023-12-30_15-01-30/epoch=15-step=10480.ckpt --target_class 2 --check_val_every_n_epoch 10 --subset_train_as_val --mode dirichlet --fid_early_stop --max_epochs 1 --validate --validate_on_test --ckpt workdir/DNA_diri_target2_valOnTrain_noDropoutEval_2024-01-08_10-11-25/epoch=739-step=242720.ckpt

python -m train_dna --run_name dirichlet_fm_target2_guidance20 --batch_size 256 --print_freq 200 --wandb --dataset_type enhancer --num_integration_steps 100 --model cnn --num_cnn_stacks 4 --cls_free_guidance --guidance_scale 20 --clean_cls_ckpt_hparams workdir/clsDNAclean_cnn_1stack_2023-12-30_15-01-30/lightning_logs/version_0/hparams.yaml --clean_cls_ckpt workdir/clsDNAclean_cnn_1stack_2023-12-30_15-01-30/epoch=15-step=10480.ckpt --target_class 2 --check_val_every_n_epoch 10 --subset_train_as_val --allow_nan_cfactor --probability_addition --fid_early_stop --validate --validate_on_test --ckpt workdir/DNA_valOnTrain_epoch12Eval_uncondFID_FIDearlyS_probAddGuidance3_2024-01-07_16-23-01/epoch=1329-step=436240.ckpt
```
