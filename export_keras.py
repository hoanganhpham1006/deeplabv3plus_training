from deeplabv4_keras import SegModel, Deeplabv3

PATH = '/vinai/hoanganh/coco2017/'
image_size = (256, 256)
bs=16
n_classes = 183
backbone = 'mobilenetv2'

SegClass = SegModel(PATH, image_size)
SegClass.set_batch_size(bs)

model = SegClass.create_seg_model(net='original', n=n_classes,\
                                      multi_gpu=True, backbone=backbone)

model.load_weights(SegClass.modelpath)
print("Loaded Multi_GPU")

single_GPU_model = model.layers[-2]
single_GPU_model.save_weights("single_GPU_seg_model.h5")
print("Saved successfully")