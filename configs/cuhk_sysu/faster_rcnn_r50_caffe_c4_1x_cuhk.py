_base_ = [
    "../_base_/faster_rcnn_r50_caffe_c4.py",
    "../_base_/cuhk_sysu.py",
    "../_base_/default_runtime.py",
]

# model settings
model = dict(roi_head=dict(bbox_head=dict(num_classes=1)))

# schedule settings
# optimizer
optimizer = dict(type="SGD", lr=0.000625, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy="step", step=[3])
# runtime settings
total_epochs = 4  # actual epoch = 4 * 3 = 12
