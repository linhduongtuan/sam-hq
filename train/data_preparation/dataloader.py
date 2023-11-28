import numpy as np
import torch.nn as nn
import torch_em
import torch_em.data.datasets as torchem_data
from torch_em.model import UNet2d
from torch_em.util.debug import check_loader, check_trainer

download_folder = "./data/livecell"
# CONFIGURE ME

# Whether to add a foreground channel (1 for all labels that are not zero) to the target.
foreground = False
# Whether to add affinity channels (= directed boundaries) or a boundary channel to the target.
# Note that you can choose at most of these two options.
affinities = False
boundaries = False

# the pixel offsets that are used to compute the affinity channels
offsets = [[-1, 0], [0, -1], [-3, 0], [0, -3], [-9, 0], [0, -9]]

assert not (affinities and boundaries), "Predicting both affinities and boundaries is not supported"

label_transform, label_transform2 = None, None
if affinities:
    label_transform2 = torch_em.transform.label.AffinityTransform(
        offsets=offsets, add_binary_target=foreground, add_mask=True
    )
elif boundaries:
    label_transform = torch_em.transform.label.BoundaryTransform(
        add_binary_target=foreground
    )
elif foreground:
    label_transform = torch_em.transform.label.labels_to_binary

patch_shape = (512, 512)
batch_size = 8

kwargs = dict(
    #ndim=2, 
    patch_shape=patch_shape, batch_size=batch_size,
    label_transform=label_transform, label_transform2=label_transform2
)

def get_dataloaders():
    train_loader = torchem_data.get_livecell_loader(download_folder, download=False, split="train", **kwargs)
    val_loader = torchem_data.get_livecell_loader(download_folder, download=False, split="val", **kwargs)

    # If you want to check several training and validation samples
    #n_samples = 4
    #print("Training samples")
    #check_loader(train_loader, n_samples, plt=True)
    #print("Validation samples")
    #check_loader(val_loader, n_samples, plt=True)

    # If you want to check shapes of images and masks on train_loader and val_loader
    for i, (images, labels) in enumerate(val_loader):
        print(type(images))
        print(f'Image shape is: {images.shape}')
        print(f'Label shape is: {labels.shape}')
    
    return train_loader, val_loader

get_dataloaders()

