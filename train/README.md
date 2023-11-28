# Training instruction for HQ-Micro-SAM

We organize the training folder as follows.
```
train
|____data
|____data_preparation
|____pretrained_checkpoint
|____train.py
|____utils
| |____dataloader.py
| |____misc.py
| |____loss_mask.py
|____segment_anything_training
|____work_dirs
```

## 1. Data Preparation

Install the `torch_em` library, I highly recommand installing the `torch_em` library using conda as follows:

```
conda install -c conda-forge torch_em
```

To prepare datasets, we utilize the `torch_em` repository for downloading and preprocessing each dataset.
If you want to load and use a single dataset for training and/or inference, follow this step:
```
python data_preration/livecell_dataloader.py
```
If you intend to train a model using a concatenated dataset, such as training a Light Microscopy generalist model, please use:
```
python data_preparation/concat_LM_training_generalist.py
```
### Expected dataset structure for HQ-Micro-SAM
```
data
|_____axondeepseg
|    |____sem
|    |____tem
|_____covid_if
|    |____train
|    |____val
|_____cremi
|_____deepbacs/mixed    
|    |____training
|    |____test
|_____dsb
|    |____train
|    |  |___images
|    |  |___masks
|    |____test
|       |___images
|       |___masks
|______hpa/hpa_dataset_v2
|    |____train
|    |____val
|    |____test
|______livecell
|    |____annotations
|    |____images
|    |____train.json
|    |____val.json
|______lucchi
|    |____lucchi_train.h5
|    |____luchi_test.h5
|______mitoEM
|    |____human_test.n5
|    |____human_train.n5
|    |____human_val.n5
|    |____rat_test.n5
|    |____rat_train.n5
|    |____rat_val.n5
|______mouse_embryo
|    |____Membrane
|    |  |___train
|    |  |___val
|    |____Nuclei
|    |  |___train
|    |  |___val
|    |____thumbnails 
|______NeuIPS22-CellSeg
|    |____Testing
|    |  |___Hidden
|    |  |___Public
|    |____TrainingUnlabeled
|    |____TrainLabeled
|    |  |___images
|    |  |___labels
|    |  |___unlabeled
|    |____Tuning
|______plantseg
|    |____nuclei_train
|    |____ovulus_train
|    |____ovulus_val
|    |____root_train
|    |____root_val
|______tissuenet
|    |____mixed
|    |   |___test
|    |   |  |___source
|    |   |  |___target
|    |   |___training
|    |   |  |___source
|    |   |  |___target
|    |____test
|    |   |___image_0000.zarr
|    |   |   |___labels
|    |   |   |___raw
|    |   ....
|    |____train
|    |   |___image_0000.zarr
|    |   |   |___labels
|    |   |   |___raw
|    |   ....
|    |____val
|    |   |___image_0000.zarr
|    |   |   |___labels
|    |   |   |___raw
|    |   .... 
```

### Expected dataset structure for HQSeg-44K
HQSeg-44K can be downloaded from [hugging face link](https://huggingface.co/sam-hq-team/sam-hq-training/tree/main/data)

```
data
|____DIS5K
|____cascade_psp
| |____DUTS-TE
| |____DUTS-TR
| |____ecssd
| |____fss_all
| |____MSRA_10K
|____thin_object_detection
| |____COIFT
| |____HRSOD
| |____ThinObject5K

```

## 2. Init Checkpoint
Init checkpoint can be downloaded from [hugging face link](https://huggingface.co/sam-hq-team/sam-hq-training/tree/main/pretrained_checkpoint)

### Expected checkpoint

```
pretrained_checkpoint
|____sam_vit_b_maskdecoder.pth
|____sam_vit_b_01ec64.pth
|____sam_vit_l_maskdecoder.pth
|____sam_vit_l_0b3195.pth
|____sam_vit_h_maskdecoder.pth
|____sam_vit_h_4b8939.pth

```

## 3. Training
#### 3.1. Training HQ-Micro-Sam: *Work in Progress*
```
python -m torch.distributed.launch --nproc_per_node=<num_gpus> train_micro_*.py --checkpoint <path/to/checkpoint> --model-type <model_type> --output <path/to/output>
```
*To train generalist models using Light Microscope datasets*
```
python -m torch.distributed.launch --nproc_per_node=8 train_micro_LM_generalist.py --checkpoint ./pretrained_checkpoint/sam_vit_b_01ec64.pth --model-type vit_b --output work_dirs/hq_sam_b_LM_generalist
```

*To train specialist models using Light Microscope datasets*
```
python -m torch.distributed.launch --nproc_per_node=8 train_micro_LM_specialist.py --checkpoint ./pretrained_checkpoint/sam_vit_b_01ec64.pth --model-type vit_b --output work_dirs/hq_sam_b_LM_specialist
```

*To train models using Electro Microscope datasets*
```
python -m torch.distributed.launch --nproc_per_node=8 train_micro_EM.py --checkpoint ./pretrained_checkpoint/sam_vit_b_01ec64.pth --model-type vit_b --output work_dirs/hq_sam_b_EM
```

#### 3.2. To train HQ-SAM on HQSeg-44K dataset

```
python -m torch.distributed.launch --nproc_per_node=<num_gpus> train.py --checkpoint <path/to/checkpoint> --model-type <model_type> --output <path/to/output>
```

*Example HQ-SAM-L training script*
```
python -m torch.distributed.launch --nproc_per_node=8 train.py --checkpoint ./pretrained_checkpoint/sam_vit_l_0b3195.pth --model-type vit_l --output work_dirs/hq_sam_l
```

*Example HQ-SAM-B training script*
```
python -m torch.distributed.launch --nproc_per_node=8 train.py --checkpoint ./pretrained_checkpoint/sam_vit_b_01ec64.pth --model-type vit_b --output work_dirs/hq_sam_b
```

*Example HQ-SAM-H training script*
```
python -m torch.distributed.launch --nproc_per_node=8 train.py --checkpoint ./pretrained_checkpoint/sam_vit_h_4b8939.pth --model-type vit_h --output work_dirs/hq_sam_h
```

### If training phase above raises an error related to distributed computing, please use `torchrun` instead of using `python -m torch.distributed.launch`:
```
torchrun --nproc_per_node=8 train.py --checkpoint ./pretrained_checkpoint/sam_vit_h_4b8939.pth --model-type vit_h --output work_dirs/hq_sam_h
```

## 4. Evaluation
### 4.1. To evaluate on 5 Light Microscope datasets: *Working in Progress*

```
torchrun --nproc_per_node=<num_gpus> train_micro_LM_*.py --checkpoint <path/to/checkpoint> --model-type <model_type> --output <path/to/output> --eval --restore-model <path/to/training_checkpoint>
```

*Example HQ-Micro-SAM-L generalist evaluation script for 5 testing Light Microscopy datasets*

```
torchrun train_micro_LM_generalist.py --checkpoint ./pretrained_checkpoint/sam_vit_l_0b3195.pth --model-type vit_l --output work_dirs/hq_micro_sam_l_generalist --eval --restore-model work_dirs/hq_micro_sam_l_generalist/epoch_11.pth
```

*Example HQ-Micro-SAM-L generalist evaluation script for 5 testing Light Microscopy datasets*

```
torchrun train_micro_LM_generalist.py --checkpoint ./pretrained_checkpoint/sam_vit_l_0b3195.pth --model-type vit_l --output work_dirs/hq_micro_sam_l_generalist --eval --restore-model work_dirs/hq_micro_sam_l_generalist/epoch_11.pth
```

*Example HQ-Micro-SAM-L specialist evaluation script for 5 testing Light Microscopy datasets*

```
torchrun train_micro_LM_generalist.py --checkpoint ./pretrained_checkpoint/sam_vit_l_0b3195.pth --model-type vit_l --output work_dirs/hq_micro_sam_l_specialist --eval --restore-model work_dirs/hq_micro_sam_l_specialist/epoch_11.pth
```

*Example HQ-Micro-SAM-L specialist visualization script for 5 testing Light Microscopy datasets*

```
torchrun --nproc_per_node=1 train_micro_LM_specialist.py --checkpoint ./pretrained_checkpoint/sam_vit_l_0b3195.pth --model-type vit_l --output work_dirs/hq_micro_sam_l_LM_specialist --eval --restore-model work_dirs/hq_micro_sam_l_LM_specialist/epoch_11.pth --visualize
```

### 4.2. To evaluate on 4 Electron Microscope datasets: *Working in Progress*

```
torchrun --nproc_per_node=<num_gpus> train_micro_EM.py --checkpoint <path/to/checkpoint> --model-type <model_type> --output <path/to/output> --eval --restore-model <path/to/training_checkpoint>
```

*Example HQ-Micro-SAM-L evaluation script for Electron Microscopy datasets*

```
torchrun --nproc_per_node=1 train_micro_EM.py --checkpoint ./pretrained_checkpoint/sam_vit_l_0b3195.pth --model-type vit_l --output work_dirs/hq_micro_sam_l_EM --eval --restore-model work_dirs/hq_micro_sam_l_EM/epoch_11.pth
```

*Example HQ-Micro-SAM-L visualization script for Electron Microscopy datasets*
```
torchrun --nproc_per_node=1 train_micro_EM.py --checkpoint ./pretrained_checkpoint/sam_vit_l_0b3195.pth --model-type vit_l --output work_dirs/hq_micro_sam_l_EM --eval --restore-model work_dirs/hq_micro_sam_l_EM/epoch_11.pth --visualize
```


### 4.3. To evaluate on 4 HQ-datasets

```
python -m torch.distributed.launch --nproc_per_node=<num_gpus> train.py --checkpoint <path/to/checkpoint> --model-type <model_type> --output <path/to/output> --eval --restore-model <path/to/training_checkpoint>
```

*Example HQ-SAM-L evaluation script*
```
python -m torch.distributed.launch --nproc_per_node=1 train.py --checkpoint ./pretrained_checkpoint/sam_vit_l_0b3195.pth --model-type vit_l --output work_dirs/hq_sam_l --eval --restore-model work_dirs/hq_sam_l/epoch_11.pth
```

*Example HQ-SAM-L visualization script*
```
python -m torch.distributed.launch --nproc_per_node=1 train.py --checkpoint ./pretrained_checkpoint/sam_vit_l_0b3195.pth --model-type vit_l --output work_dirs/hq_sam_l --eval --restore-model work_dirs/hq_sam_l/epoch_11.pth --visualize
```

> Mostly borrowed this work from [**Segment Anything in High Quality**](https://arxiv.org/abs/2306.01567)           
> Lei Ke, Mingqiao Ye, Martin Danelljan, Yifan Liu, Yu-Wing Tai, Chi-Keung Tang, Fisher Yu \
> ETH Zurich & HKUST
> 
