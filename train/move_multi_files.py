import shutil
import glob
import os

source_folder = glob.glob("/Users/linh/Downloads/cv/torch-em/experiments/training_data/livecell/annotations/livecell_train_val_images/*/*.tif")
destination_folder = "/Users/linh/Downloads/cv/torch-em/experiments/training_data/livecell/gt/"
os.mkdir(destination_folder)
#source_folder = "/Users/linh/Downloads/cv/torch-em/experiments/training_data/livecell/annotations/livecell_train_val_images/"


# fetch all files
#for file_name in os.listdir(source_folder):
for file_name in source_folder:
    print(file_name)
    shutil.copy2(file_name, destination_folder)
    # construct full file path
    #source = source_folder + file_name
    #destination = destination_folder + file_name
    # move only files
    #if os.path.isfile(file_name):
    #    shutil.copy(file_name, destination)
    #    print('Moved:', file_name)

"""
import os
import shutil

images = [f for f in os.listdir("/Users/linh/Downloads/cv/torch-em/experiments/training_data/livecell/annotations/livecell_train_val_images/") if '.tif' in f.lower()]

os.mkdir('/Users/linh/Downloads/cv/torch-em/experiments/training_data/livecell/gt')

for image in images:
    new_path = '/Users/linh/Downloads/cv/torch-em/experiments/training_data/livecell/gt' + image
    #shutil.move(image, new_path)
    shutil.copy2(image, new_path)
    
 """ 
"""
import os
import pathlib

images = [f for f in os.listdir("/Users/linh/Downloads/cv/torch-em/experiments/training_data/livecell/annotations/livecell_train_val_images/") if '.tif' in f.lower()]

os.mkdir('/Users/linh/Downloads/cv/torch-em/experiments/training_data/livecell/annotations/livecell_train_val_images/')

for image in images:
    new_path = '/Users/linh/Downloads/cv/torch-em/experiments/training_data/livecell/gt' + image
    pathlib.Path(image).rename(new_path)
    """
"""
import shutil
import os
    
source_dir = "/Users/linh/Downloads/cv/torch-em/experiments/training_data/livecell/annotations/livecell_train_val_images/"
target_dir = '/Users/linh/Downloads/cv/torch-em/experiments/training_data/livecell/gt'
print(images)    
file_names = os.listdir(source_dir)
print(file_names)    
#for file_name in file_names:
#    shutil.move(os.path.join(source_dir, file_name), target_dir)
"""    