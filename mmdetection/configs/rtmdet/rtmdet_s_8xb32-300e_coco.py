_base_ = './rtmdet_l_8xb32-300e_coco.py'
# 数据集类型及路径
dataset_type = 'CocoDataset'
# data_root = 'mmdetection/data/coco-2017/'  
data_root = 'data/coco-person-limited/'  

# metainfo = {'classes': ('sjb_rect',)}
# NUM_CLASSES = len(metainfo['classes'])
metainfo = {'classes': ('person',)}  # 更新类别为 person  
NUM_CLASSES = len(metainfo['classes'])  # 现在 NUM_CLASSES 将为 1  
# NUM_CLASSES = 80


checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth'  # noqa

# 训练超参数
MAX_EPOCHS = 50
TRAIN_BATCH_SIZE = 2
VAL_BATCH_SIZE = 2
VAL_INTERVAL = 5 # 每隔多少轮评估保存一次模型权重
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=MAX_EPOCHS, val_interval=VAL_INTERVAL)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

backend_args = None

model = dict(
    backbone=dict(
        deepen_factor=0.33,
        widen_factor=0.5,
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    neck=dict(in_channels=[128, 256, 512], out_channels=128, num_csp_blocks=1),
    bbox_head=dict(in_channels=128, feat_channels=128, exp_on_reg=False))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),
    dict(type='PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
# DataLoader
train_dataloader = dict(
    batch_size=TRAIN_BATCH_SIZE,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        # ann_file='annotations/person_keypoints_train2017.json',
        ann_file='train_annotations.coco.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=VAL_BATCH_SIZE,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        # ann_file='annotations/person_keypoints_val2017.json',
        ann_file='valid_annotations.coco.json',
        data_prefix=dict(img='valid/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=280,
        switch_pipeline=train_pipeline_stage2)
]
