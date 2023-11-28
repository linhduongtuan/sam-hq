from torch.utils.data import DataLoader, ConcatDataset
from torch_em.data.datasets.deepbacs import get_deepbacs_dataset
from torch_em.data.datasets.livecell import get_livecell_dataset
from torch_em.data.datasets.tissuenet import get_tissuenet_dataset

patch_shape=(512, 512)

deepbacs_train_dataset = get_deepbacs_dataset("./data/deepbacs/",
                                              patch_shape=patch_shape,
                                              split="train")
deepbacs_val_dataset = get_deepbacs_dataset("./data/deepbacs/", 
                                              patch_shape=patch_shape,
                                              split="test")

livecell_train_dataset = get_livecell_dataset("./data/livecell", split="train", patch_shape=patch_shape)
livecell_val_dataset = get_livecell_dataset("./data/livecell", split="val", patch_shape=patch_shape)

tissuenet_train_dataset = get_tissuenet_dataset("./data/tissuenet/",  
                                                raw_channel="nucleus", #"cell", "rgb",
                                                label_channel="nucleus", #"cell",
                                                patch_shape=patch_shape,
                                                split="train")
tissuenet_val_dataset = get_tissuenet_dataset("./data/tissuenet",
                                              raw_channel="nucleus", #"cell", "rgb",
                                              label_channel="nucleus", #"cell", 
                                              patch_shape=patch_shape,
                                              split="test")

concat_train_datasets = ConcatDataset([deepbacs_train_dataset, 
                                       #dsb_train_dataset, 
                                       livecell_train_dataset, 
                                       #neurips_cell_seg_train_dataset,
                                       #plantseg_train_data,
                                       tissuenet_train_dataset 
                                       ])
concat_val_datasets = ConcatDataset([deepbacs_val_dataset, 
                                       #dsb_val_dataset, 
                                       livecell_val_dataset, 
                                       #neurips_cell_seg_val_dataset,
                                       #plantseg_val_data,
                                       tissuenet_val_dataset 
                                       ])

train_datasets_loaders = DataLoader(dataset=concat_train_datasets, batch_size=8)
val_datasets_loaders = DataLoader(dataset=concat_val_datasets, batch_size=8)


image, mask = next(iter(val_datasets_loaders))
print(f'Image shape is {image.shape}')
print(f'Mask shape is {mask.shape}')
assert image.shape == (8, 1, 512, 512)