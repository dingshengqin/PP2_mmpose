_base_ = ['../../../_base_/default_runtime.py']

# runtime
max_epochs = 420
stage2_num_epochs = 30
base_lr = 4e-3
# metainfo = {'classes': ('person',),
#             'dataset_name':'CocoDataset',
#             'keypoint_info':{
#                 0:
#                 dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
#                 1:
#                 dict(
#                     name='left_eye',
#                     id=1,
#                     color=[51, 153, 255],
#                     type='upper',
#                     swap='right_eye'),
#                 2:
#                 dict(
#                     name='right_eye',
#                     id=2,
#                     color=[51, 153, 255],
#                     type='upper',
#                     swap='left_eye'),
#                 3:
#                 dict(
#                     name='left_ear',
#                     id=3,
#                     color=[51, 153, 255],
#                     type='upper',
#                     swap='right_ear'),
#                 4:
#                 dict(
#                     name='right_ear',
#                     id=4,
#                     color=[51, 153, 255],
#                     type='upper',
#                     swap='left_ear'),
#                 5:
#                 dict(
#                     name='left_shoulder',
#                     id=5,
#                     color=[0, 255, 0],
#                     type='upper',
#                     swap='right_shoulder'),
#                 6:
#                 dict(
#                     name='right_shoulder',
#                     id=6,
#                     color=[255, 128, 0],
#                     type='upper',
#                     swap='left_shoulder'),
#                 7:
#                 dict(
#                     name='left_elbow',
#                     id=7,
#                     color=[0, 255, 0],
#                     type='upper',
#                     swap='right_elbow'),
#                 8:
#                 dict(
#                     name='right_elbow',
#                     id=8,
#                     color=[255, 128, 0],
#                     type='upper',
#                     swap='left_elbow'),
#                 9:
#                 dict(
#                     name='left_wrist',
#                     id=9,
#                     color=[0, 255, 0],
#                     type='upper',
#                     swap='right_wrist'),
#                 10:
#                 dict(
#                     name='right_wrist',
#                     id=10,
#                     color=[255, 128, 0],
#                     type='upper',
#                     swap='left_wrist'),
#                 11:
#                 dict(
#                     name='left_hip',
#                     id=11,
#                     color=[0, 255, 0],
#                     type='lower',
#                     swap='right_hip'),
#                 12:
#                 dict(
#                     name='right_hip',
#                     id=12,
#                     color=[255, 128, 0],
#                     type='lower',
#                     swap='left_hip'),
#                 13:
#                 dict(
#                     name='left_knee',
#                     id=13,
#                     color=[0, 255, 0],
#                     type='lower',
#                     swap='right_knee'),
#                 14:
#                 dict(
#                     name='right_knee',
#                     id=14,
#                     color=[255, 128, 0],
#                     type='lower',
#                     swap='left_knee'),
#                 15:
#                 dict(
#                     name='left_ankle',
#                     id=15,
#                     color=[0, 255, 0],
#                     type='lower',
#                     swap='right_ankle'),
#                 16:
#                 dict(
#                     name='right_ankle',
#                     id=16,
#                     color=[255, 128, 0],
#                     type='lower',
#                     swap='left_ankle')
#             },
#             'skeleton_info':{
#                 0:
#                 dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
#                 1:
#                 dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
#                 2:
#                 dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
#                 3:
#                 dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
#                 4:
#                 dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
#                 5:
#                 dict(link=('left_shoulder', 'left_hip'), id=5, color=[51, 153, 255]),
#                 6:
#                 dict(link=('right_shoulder', 'right_hip'), id=6, color=[51, 153, 255]),
#                 7:
#                 dict(
#                     link=('left_shoulder', 'right_shoulder'),
#                     id=7,
#                     color=[51, 153, 255]),
#                 8:
#                 dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
#                 9:
#                 dict(
#                     link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
#                 10:
#                 dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
#                 11:
#                 dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
#                 12:
#                 dict(link=('left_eye', 'right_eye'), id=12, color=[51, 153, 255]),
#                 13:
#                 dict(link=('nose', 'left_eye'), id=13, color=[51, 153, 255]),
#                 14:
#                 dict(link=('nose', 'right_eye'), id=14, color=[51, 153, 255]),
#                 15:
#                 dict(link=('left_eye', 'left_ear'), id=15, color=[51, 153, 255]),
#                 16:
#                 dict(link=('right_eye', 'right_ear'), id=16, color=[51, 153, 255]),
#                 17:
#                 dict(link=('left_ear', 'left_shoulder'), id=17, color=[51, 153, 255]),
#                 18:
#                 dict(
#                     link=('right_ear', 'right_shoulder'), id=18, color=[51, 153, 255])
#             },

#             'joint_weights':[
#                 1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5,
#                 1.5
#             ],
#             'sigmas':[
#                 0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
#                 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
#             ]
            
#             }  # 更新类别为 person  
metainfo = {
    'classes': ('person',),
    "dataset_name":'CocoDataset',
    "paper_info":dict(
        author='Lin, Tsung-Yi and Maire, Michael and '
        'Belongie, Serge and Hays, James and '
        'Perona, Pietro and Ramanan, Deva and '
        r'Doll{\'a}r, Piotr and Zitnick, C Lawrence',
        title='Microsoft coco: Common objects in context',
        container='European conference on computer vision',
        year='2014',
        homepage='http://cocodataset.org/',
    ),
    "keypoint_info":{
        0:
        dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='left_eye',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='right_eye'),
        2:
        dict(
            name='right_eye',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='left_eye'),
        3:
        dict(
            name='left_ear',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='right_ear'),
        4:
        dict(
            name='right_ear',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='left_ear'),
        5:
        dict(
            name='left_shoulder',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        6:
        dict(
            name='right_shoulder',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        7:
        dict(
            name='left_elbow',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        8:
        dict(
            name='right_elbow',
            id=8,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        9:
        dict(
            name='left_wrist',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        10:
        dict(
            name='right_wrist',
            id=10,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        11:
        dict(
            name='left_hip',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        12:
        dict(
            name='right_hip',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        13:
        dict(
            name='left_knee',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        14:
        dict(
            name='right_knee',
            id=14,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        15:
        dict(
            name='left_ankle',
            id=15,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        16:
        dict(
            name='right_ankle',
            id=16,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle')
    },
    "skeleton_info":{
        0:
        dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('left_shoulder', 'left_hip'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('right_shoulder', 'right_hip'), id=6, color=[51, 153, 255]),
        7:
        dict(
            link=('left_shoulder', 'right_shoulder'),
            id=7,
            color=[51, 153, 255]),
        8:
        dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
        9:
        dict(
            link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('left_eye', 'right_eye'), id=12, color=[51, 153, 255]),
        13:
        dict(link=('nose', 'left_eye'), id=13, color=[51, 153, 255]),
        14:
        dict(link=('nose', 'right_eye'), id=14, color=[51, 153, 255]),
        15:
        dict(link=('left_eye', 'left_ear'), id=15, color=[51, 153, 255]),
        16:
        dict(link=('right_eye', 'right_ear'), id=16, color=[51, 153, 255]),
        17:
        dict(link=('left_ear', 'left_shoulder'), id=17, color=[51, 153, 255]),
        18:
        dict(
            link=('right_ear', 'right_shoulder'), id=18, color=[51, 153, 255])
    },
    "joint_weights":[
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5,
        1.5
    ],
    "sigmas":[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
    ]
}
# metainfo = {'classes': ('person',)} 

train_cfg = dict(max_epochs=max_epochs, val_interval=10)
randomness = dict(seed=21)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 210 to 420 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=1024)

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=(416, 416),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.167,
        widen_factor=0.375,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
            'rtmposev1/cspnext-tiny_udp-aic-coco_210e-256x192-cbed682d_20230130.pth'  # noqa
        )),
    head=dict(
        type='RTMCCHead',
        in_channels=384,
        out_channels=17,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=True))

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/coco-2017/'  


backend_args = dict(backend='local')
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         f'{data_root}': 's3://openmmlab/datasets/detection/coco/',
#         f'{data_root}': 's3://openmmlab/datasets/detection/coco/'
#     }))

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.6, 1.4], rotate_factor=80),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.75, 1.25],
        rotate_factor=60),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=256,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='person_keypoints_val2017.json',
        data_prefix=dict(img='val_images/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=64,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        data_mode=data_mode,
        ann_file='person_keypoints_val2017.json',
        # bbox_file=f'{data_root}person_detection_results/'
        # 'COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='val_images/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(save_best='coco/AP', rule='greater', max_keep_ckpts=1))

custom_hooks = [
    # Turn off EMA while training the tiny model
    # dict(
    #     type='EMAHook',
    #     ema_type='ExpMomentumEMA',
    #     momentum=0.0002,
    #     update_buffers=True,
    #     priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'person_keypoints_val2017.json')
test_evaluator = val_evaluator
