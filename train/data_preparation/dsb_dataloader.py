import torch_em.data.datasets as torchem_data

data_path = "./data/dsb"

patch_shape = (512, 512)
def get_dsb_dataloaders(path):
    train_loader = torchem_data.get_dsb_loader(path=data_path, 
                                                    patch_shape=patch_shape,
                                                    batch_size=8,
                                                    download=False,
                                                    split='train')
    val_loader = torchem_data.get_dsb_loader(path=data_path, 
                                                  patch_shape=patch_shape,
                                                  batch_size=8,
                                                  download=False,
                                                  split='test')
    return train_loader, val_loader

train_loader, val_loader = get_dsb_dataloaders(path=data_path)
image, mask = next(iter(val_loader))
print(f'Image shape is {image.shape}')
print(f'Mask shape is {mask.shape}')
assert image.shape == (8, 1, 512, 512)



