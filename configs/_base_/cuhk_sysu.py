# dataset settings
dataset_type = "CUHKSYSU"
data_root = "data/cuhk_sysu"
# use caffe img_norm
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", img_scale=(1000, 600), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="LoadPersonID"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "gt_pids"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadProposals"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img", "proposals"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type="RepeatDataset",
        times=3,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            split="train",
            ann_file=None,
            pipeline=train_pipeline,
        ),
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        split="test",
        ann_file=None,
        proposal_file="",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        split="test",
        ann_file=None,
        proposal_file="",
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=1, metric="mAP")
