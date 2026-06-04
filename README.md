## Fork Changes

**Data & Dataset Pipeline**
- Added DataComp-medium and CC12M dataset support with automated download scripts
- Optimized CC12M extraction pipeline: sequential tar streaming with JPEG passthrough (avoids costly re-encoding)
- Added ImageNet-1k download/extraction utilities and evaluation dataset fetching scripts
- Added precompute modules for ImageNet and CC12M to cache CLIP/DINOv2 embeddings on disk, eliminating redundant forward passes during training

**Training Infrastructure**
- Implemented multi-GPU DDP training with configurable gradient checkpointing
- Added bf16 mixed precision training via `--bf16` flag with automatic GradScaler handling
- Increased DDP `bucket_cap_mb` to 200MB for better computation/communication overlap on high-bandwidth interconnects
- Added TensorBoard logging with per-step metrics and CSV export for offline analysis
- Fixed DataLoader crash on truncated/corrupt images by setting `LOAD_TRUNCATED_IMAGES` in worker init
- Fixed training resume: DynamicPenaltyWeight state (current weight + EMA) now persisted in checkpoints
- Fixed loss EMA percentage spike at epoch boundaries by carrying EMA state across epochs

**Training Algorithms & Hyperparameters**
- Added dynamic penalty weight scheduling with optional cosine decay (`--dynamic_pw_cosine_decay`)
- Added cap alignment mechanism to bound embedding drift during fine-tuning
- Added `clean_weight` as a configurable experiment parameter for controlling clean loss contribution
- Added sparsity computation for analyzing embedding activation patterns
- Configured multiple experiment sweeps: dpw_target ∈ {0.5, 0.8, 1.0}, lr ∈ {8e-5, 1e-4, 2e-4}, cosine decay on/off

**Metrics & Monitoring**
- Track per-batch embedding variance (`emb_var` local, `emb_var_global` via all-reduce) in TensorBoard and CSV
- Added embedding effective rank (`emb_erank`) metric computed via SVD entropy — measures dimensionality utilization
- Added `ClipDriftMetric`: monitors L2 and angular drift of fine-tuned embeddings vs frozen CLIP baseline on a fixed validation batch
- Fixed drift metrics not appearing in TensorBoard due to missing `tb_writer` calls in eval loop

**Checkpointing & Experiment Management**
- Save step checkpoints at 25/50/75/100% progress per epoch (8 total for 2-epoch runs)
- Skip redundant `final.pt` when epoch checkpoints are enabled
- Unified experiment naming via `train/naming.py` — single source of truth for directory names across train/eval scripts
- Made `EXPERIMENTS_DIR` and `EVAL_BASE` overridable via environment variables for portability across machines

**Evaluation**
- Added `run_eval_checkpoints.sh` with rolling GPU allocation for parallel evaluation of all checkpoints
- Added `make_tables.py` for generating comparison tables of eval results vs OpenAI CLIP baseline (supports multiple experiments as columns, flat experiment dirs, argparse interface)
- Batch size benchmark documentation for ViT-L-14 on H100 (including explanation of flat throughput at high batch sizes)

**LLaVA Integration**
- Added LLaVA LoRA fine-tuning script with KUEA vision tower support
- Added data download script for LLaVA v1.5 mix665k training data
- Added parallel LLaVA evaluation script with configurable task groups and GPU assignment
- Integrated lmms-eval framework with KUEA-specific compatibility patches (tracked as source files, not submodule)
- Fixed Gradio 6.x compatibility and LLaVA v1.5 inference with transformers 4.37+
- Fixed critical inference bugs: DynamicCache initialization and KUEA vision tower weight loading
- Optimized eval throughput 24x via thread limiting + parallel image preprocessing (BS=32, max_new_tokens=64 for refcoco)

**Environment**
- Added conda environment exports pinned to torch 2.0.1+cu118
- Updated `requirements_frozen.txt` with all training dependencies

----

# Kernel-based Unsupervised Embedding Alignment for Enhanced Visual Representation in Vision-language Models
Implementation for ICML 2025 paper [Kernel-based Unsupervised Embedding Alignment for Enhanced Visual Representation 
in Vision-language Models](https://arxiv.org/abs/2506.02557)
by [Shizhan Gong](https://peterant330.github.io/), Yankai Jiang, [Qi Dou](https://www.cse.cuhk.edu.hk/~qdou/), 
and [Farzan Farnia](https://www.cse.cuhk.edu.hk/~farnia/)

<img align="center" src="asset/method.png" width="750">

## Setup
We recommend to install the environment through conda:

```
cd KUEA
conda create --name myenv python=3.11
conda activate myenv
pip install -r requirements.txt
```

## Alignment Fine-tuning
Please use the following code for the alignment fine-tuning.

```commandline
python -m train.align_training_clip --clip_model_name ViT-L-14 --pretrained openai --dataset imagenet 
--imagenet_root /path/to/imagenet2012 --template std --output_normalize False --steps 40000 --warmup 2800 
--batch_size 64 --loss l2 --loss_clean l2 --opt adamw --lr 1e-5 --wd 1e-4 --inner_loss l2 --wandb False 
--output_dir /path/to/checkpoint --clean_weight 1. --penalty_weight 0.5 --kernel_dino polynomial 
--kernel_clip polynomial --gamma 0.0032 --coef0 0.191623 --experiment_name exp_1  --log_freq 1 --eval_freq 10
```

`--imagenet_root` should be adjusted to designate the directory of the imagenet dataset. `--output_dir` specifies the
directory to store the fine-tuned checkpoint. `--gamma` and `--coef0` are the initial parameters used to calculate the
polynomial kernel of CLIP representations. We pre-calculate them by sampling several images from the training data and
minimize the L2 distance between kernel matrices of CLIP and DINOv2.

## Evaluation
We utilize [CLIP-Benchmark](https://github.com/LAION-AI/CLIP_benchmark) for evaluation of the fine-tuned models.

To evaluate the model, first go to the `CLIP_benchmark` directory

```
cd CLIP_benchmark
```

Edit the file `benchmark/models.txt` to include the model to evaluate:

```commandline
ViT-L-14-336,openai
ViT-L-14-336,directory/to/finetuned/models.pt
```
The first element specify the architecture of the model, and the second element specify the saved checkpoints. Using 
`openai` for evaluation of the original CLIP model. Then run the corresponding bash command:
```commandline
./bash/run_benchmark_clean.sh # zero-shot classification
./bash/run_benchmark_lp.sh # linear probing
./bash/run_benchmark_rt.sh # image-text retrieval
```
Please edit the `SAVE_DIR` field of the corresponding files, which specifies the directory to save the evaluation results.

## Fine-tuning of LLaVA
The script to fine-tune LLaVA is adjusted from [LLaVA](https://github.com/haotian-liu/LLaVA). We use the following
command to perform LoRA fine-tuning
```commandline
cd LLaVA
./scripts/v1_5/finetune_task_lora.sh
```
Note to edit the `--vision_tower` filed of the script to denote the directory of the checkpoints after the alignment fine-tuning.

## Evaluation of LLaVA
We utilize the tool provided by [Prismatic library](https://github.com/TRI-ML/prismatic-vlms) for evaluation of the LLaVA.

## Pre-trained checkpoints
The pretrained checkpoints for the CLIP vision encoder can be downloaded from OneDrive.

[ViT-L-14-224](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155187960_link_cuhk_edu_hk/Evj3UUqXLpRNjwQ0pQi-NugB7-JuKxU4xxGiqjrBH_MRDA?e=7SNAec)

[ViT-L-14-336](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155187960_link_cuhk_edu_hk/Eh90Ji9PvF9Hk70NEa0pKcsBReM1UDIVm3fTNUNKB6pngQ?e=57emeO)

## Bibtex
If you find this work helpful, you can cite our paper as follows:
```commandline
@article{gong2025kernel,
  title={Kernel-based Unsupervised Embedding Alignment for Enhanced Visual Representation in Vision-language Models},
  author={Gong, Shizhan and Jiang, Yankai and Dou, Qi and Farnia, Farzan},
  journal={arXiv preprint arXiv:2506.02557},
  year={2025}
}
```

## Contact
For any questions, please contact [szgong22@cse.cuhk.edu.hk](szgong22@cse.cuhk.edu.hk)


