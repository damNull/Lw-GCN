# Evaluations of Lw-GCN

The command for evaluate models.

Unzip the files from `https://drive.google.com/file/d/1EwAZWIPSpX2WTYVBwvEoebp0nLBRYbaY/view?usp=sharing` into `evaluations/` directory for model evaluations.

The calculation complexity of our single stream model is less than 1 GFLOPs which is suitable for edge / mobile device.

## Ablation study

### Shift-GCN baseline

`python eval.py --test-dir ./evaluations/NTU-RGBD/ablation_study/Shift-GCN`

The performance of models:
Name|Top1(%)
-|-
Shift-GCN|87.73

### Shift-GCN tiny model with Shift-GCN distillation (using KD)

The training time of distilled tiny Shift-GCN model is ~2.5x than directly training Lw-GCN.

`python eval.py --test-dir ./evaluations/NTU-RGBD/ablation_study/Shift-GCN_Shift-GCN-S`

The performance of models:
Name|Top1(%)
-|-
Shift-GCN-S (KD)|87.35

### Contrastive learning

#### Use 2 args rotation augmentation for contrastive learning

`python eval.py --test-dir ./evaluations/NTU-RGBD/ablation_study/Contrastive_Learning/Aug_2arg_rot`

#### Use random scale and totation augmentation for contrastive learning

`python eval.py --test-dir ./evaluations/NTU-RGBD/ablation_study/Contrastive_Learning/Aug_rand_scale_rot`

#### Use random shear and totation augmentation for contrastive learning

`python eval.py --test-dir ./evaluations/NTU-RGBD/ablation_study/Contrastive_Learning/Aug_rand_shear_rot`

The performance of models:
Name|Top1(%)
-|-
Aug_2arg_rot|87.89
Aug_rand_scale_rot|87.48
Aug_rand_shear_rot|86.28

## Final model

### NTU-RGB+D

#### xview

`python eval.py --test-dir ./evaluations/NTU-RGBD/Lw-GCN/xview/<eval_stream>`

> <eval_stream> can be "joint", "joint_motion", "bone" and "bone_motion".

The performance of models:
Name|Top1(%)
-|-
joint|94.07
joint_motion|92.77
bone|94.25
bone_motion|92.45

#### xsub

`python eval.py --test-dir ./evaluations/NTU-RGBD/Lw-GCN/xsub/<eval_stream>`

> <eval_stream> can be "joint", "joint_motion", "bone" and "bone_motion".

The performance of models:
Name|Top1(%)
-|-
joint|87.89
joint_motion|86.20
bone|87.86
bone_motion|86.08

### NTU-RGB+D 120

#### xsetup

`python eval.py --test-dir ./evaluations/NTU-RGBD120/Lw-GCN/xsetup/<eval_stream>`

> <eval_stream> can be "joint", "joint_motion", "bone" and "bone_motion".

The performance of models:
Name|Top1(%)
-|-
joint|82.44
joint_motion|80.03
bone|83.63
bone_motion|80.33

#### xsub

`python eval.py --test-dir ./evaluations/NTU-RGBD120/Lw-GCN/xsub/<eval_stream>`

> <eval_stream> can be "joint", "joint_motion", "bone" and "bone_motion".

The performance of models:
Name|Top1(%)
-|-
joint|80.91
joint_motion|78.26
bone|83.05
bone_motion|79.15

### Northwestern-UCLA

`python eval.py --test-dir ./evaluations/NW-UCLA/<eval_stream>`

> <eval_stream> can be "joint", "joint_motion", "bone" and "bone_motion".

The performance of models:
Name|Top1(%)
-|-
joint|90.73
joint_motion|93.53
bone|88.58
bone_motion|89.01

## Ensemble

After the execution of evaluation, the script will save the result at the model dir. Execute the scripts at `toos/` can get ensemble results.

