from torch.utils.data import DataLoader, ConcatDataset
from torch_em.data.datasets.covid_if import get_covid_if_dataset
from torch_em.data.datasets.plantseg import get_plantseg_dataset
from torch_em.data.datasets.hpa import get_hpa_segmentation_dataset
from torch_em.data.datasets.lizard import get_lizard_dataset
from torch_em.data.datasets.mouse_embryo import get_mouse_embryo_dataset



patch_shape=(512, 512)

covid_if_testing_dataset = get_covid_if_dataset("/Users/linh/Downloads/cv/torch-em/experiments/training_data/covid_if", patch_shape=patch_shape)

#hpa_testing_dataset = get_hpa_segmentation_dataset("/Users/linh/Downloads/cv/torch-em/experiments/training_data/hpa", split="test", patch_shape=patch_shape )

#lizard_testing_dataset = get_lizard_dataset("/Users/linh/Downloads/cv/torch-em/experiments/training_data/lizard", patch_shape=patch_shape)

mouse_embryo_testing_dataset = get_mouse_embryo_dataset("/Users/linh/Downloads/cv/torch-em/experiments/training_data/mouse_embryo",
                                                        name="membrane", # "nuclei"
                                                        split="val",
                                                        patch_shape=(1, 512, 512))

plantseg_testing_dataset = get_plantseg_dataset(path="/Users/linh/Downloads/cv/torch-em/experiments/training_data/plantseg", patch_shape=(1, 512, 512), name="ovules", split="val")



concat_testing_datasets = ConcatDataset([covid_if_testing_dataset,
                                       #hpa_testing_dataset,
                                       #lizard_testing_dataset,
                                       mouse_embryo_testing_dataset,
                                       plantseg_testing_dataset,
])
                                       

#val_loader = ConcatDataset([dsb_train_dataset, livecell_val_dataset])
testing_datasets_loaders = DataLoader(dataset=concat_testing_datasets, batch_size=8)


image, mask = next(iter(testing_datasets_loaders))
print(f'Image shape is {image.shape}')
print(f'Mask shape is {mask.shape}')
assert image.shape == (8, 1, 512, 512)