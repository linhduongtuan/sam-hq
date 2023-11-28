import torch_em.data.datasets as torchem_data

data_path = "./data/lucchi"

patch_shape = (3, 512, 512)
def get_lucchi_dataloaders(path):
    train_loader = torchem_data.get_lucchi_loader(path, 
                                                    patch_shape=patch_shape,
                                                    batch_size=8,
                                                    download=False,
                                                    split='train')
    val_loader = torchem_data.get_lucchi_loader(path, 
                                                  patch_shape=patch_shape,
                                                  batch_size=8,
                                                  download=False,
                                                  split='test')
    return train_loader, val_loader

train_loader, val_loader = get_lucchi_dataloaders(path=data_path)
image, mask = next(iter(val_loader))
print(f'Image shape is {image.shape}')
print(f'Mask shape is {mask.shape}')
assert image.shape == (8, 1, 3, 512, 512)



