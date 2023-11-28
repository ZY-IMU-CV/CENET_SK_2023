_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
] 
model = dict(
    backbone=dict(
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        multi_grid=(1, 2, 4)
    ),
    decode_head=dict(
        num_classes=150,
        channels=512,
        dilations=(1, 6, 12, 18),
        c1_in_channels=256,
        c1_channels=256,
        sampler=dict(type='OHEMPixelSampler', min_kept=100000)
    ),
    auxiliary_head=dict(num_classes=150)
)
data = dict(
    samples_per_gpu=8
)
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)