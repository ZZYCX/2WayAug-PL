from mlcpl import dataset
import torchvision

# config the dataset here

#单卡4090
train_dataset = dataset.MSCOCO('/home/sx639/GZS/coco2014/', split='train').drop_labels_random(0.1)
valid_dataset = dataset.MSCOCO('/home/sx639/GZS/coco2014/', split='valid')
#多卡4090
# train_dataset = dataset.MSCOCO('/media/ubuntu2/A/coco2014', split='train').drop_labels_random(0.1)
# valid_dataset = dataset.MSCOCO('/media/ubuntu2/A/coco2014', split='valid')

## example: VG-200 with label proportion 10%
# train_dataset = dataset.VG_200('/Users/max/Documents/datasets/VG', split='train').drop_labels_random(0.1)
# valid_dataset = dataset.VG_200('/Users/max/Documents/datasets/VG', split='valid')

## example: VOC 2007 with label proportion 90%
# train_dataset = dataset.Pascal_VOC_2007('/Users/max/Documents/datasets/VOC2007', split='train').drop_labels_random(0.9)
# valid_dataset = dataset.Pascal_VOC_2007('/Users/max/Documents/datasets/VOC2007', split='valid')


image_size = (448, 448)

batch_size = 32
accum_step = 4
lr = 2e-4
epochs = 60
early_stopping = 30
ema = 0.999
weight_decay = 1e-4

# logicmix
probability = 0.5
num_samples = (2, 4) # [inclusive, exclusive)

# DP-ASL
gamma_p, gamma_n, m = 0, 4, 0.05
omega_p, omega_n, n = 4, 0, 0   # if they are set as the same as above variables, it becomes ordinal asymmetric loss

# Pseudo-labeling
thresholds = (-4, 4)
E = 20

# Conventional Augmentation
data_aug = (
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandAugment(interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR),
    # torchvision.transforms.LinearTransformation(),
)

device = 'cuda:1'
