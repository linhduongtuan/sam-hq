from torch.utils.data import DataLoader, ConcatDataset
from torch_em.data.datasets.cremi import get_cremi_dataset
from torch_em.data.datasets.axondeepseg import get_axondeepseg_dataset
from torch_em.data.datasets.mitoem import get_mitoem_dataset



patch_shape_2=(512, 512)
patch_shape_3=(1, 512, 512)

cremi_train_dataset = get_cremi_dataset("./data/cremi",  patch_shape=patch_shape_3)
cremi_val_dataset = get_cremi_dataset("./data/cremi",  patch_shape=patch_shape_3)

#hpa_testing_dataset = get_hpa_segmentation_dataset("/Users/linh/Downloads/cv/torch-em/experiments/training_data/hpa", split="test", patch_shape=patch_shape )

#lizard_testing_dataset = get_lizard_dataset("/Users/linh/Downloads/cv/torch-em/experiments/training_data/lizard", patch_shape=patch_shape)

axondeepseg_train_dataset = get_axondeepseg_dataset("./data/axondeepseg",
                                                       download=True, 
                                                       name="sem", #"tem"
                                                       split="train",
                                                       patch_shape=patch_shape_3)
axondeepseg_val_dataset = get_axondeepseg_dataset("./data/axondeepseg",
                                                       download=True, 
                                                       name="sem", #"tem"
                                                        split="val",
                                                        patch_shape=patch_shape_3)   

mitoem_train_dataset = get_mitoem_dataset("./data/mitoem",
                                                       download=True, # "nuclei"
                                                        splits="train",
                                                        patch_shape=patch_shape_3)
mitoem_val_dataset = get_mitoem_dataset("./data/mitoem",
                                                       download=True, # "nuclei"
                                                        splis="val",
                                                        patch_shape=patch_shape_3)                                                     


#platyem_train_dataset = get_platyem_dataset("/Users/linh/Downloads/cv/torch-em/experiments/training_data/platyem",
#                                                       download=True, # "nuclei"
#                                                        split="train",
#                                                        patch_shape=patch_shape_3)
#platyemem_val_dataset = get_platyem_dataset("/Users/linh/Downloads/cv/torch-em/experiments/training_data/platyem",
#                                                       download=True, # "nuclei"
#                                                        split="val",
#                                                        patch_shape=patch_shape_3) 



concat_train_datasets = ConcatDataset([cremi_train_dataset,
                                       axondeepseg_train_dataset,
                                       mitoem_train_dataset,
                                       #platyemem_val_dataset                                       
])
                                       
concat_val_datasets = ConcatDataset([cremi_val_dataset,
                                       axondeepseg_val_dataset,
                                       mitoem_val_dataset,
                                       #platyemem_val_dataset
                                       
])

train_datasets_loaders = DataLoader(dataset=concat_train_datasets, batch_size=8)
val_datasets_loaders = DataLoader(dataset=concat_val_datasets, batch_size=8)


image, mask = next(iter(val_datasets_loaders))
print(f'Image shape is {image.shape}')
print(f'Mask shape is {mask.shape}')
assert image.shape == (8, 1, 512, 512)