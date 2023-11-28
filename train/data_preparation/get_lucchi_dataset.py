
import torch_em

from torch_em.data.datasets.lucchi import get_lucchi_loader 


download_path = "./data/LUCHI"
patch_shape = (1024, 1024)
def get_lucchi_dataloaders():
    train_loader = get_lucchi_loader(download_path, patch_shape=patch_shape, batch_size=1, download=True, split="train")
    val_loader = get_lucchi_loader(download_path, patch_shape=patch_shape, batch_size=1, download=True, split="test")
    print(f'Length of training set is {len(train_loader)}')
    print(f'Length of testing set is {len(val_loader)}')
    return train_loader, val_loader

get_lucchi_dataloaders()

