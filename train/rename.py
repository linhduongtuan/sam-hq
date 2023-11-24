import glob
import os

path = "/proj/aicell/users/linh/cv/sam-hq/train/data/Lucchi/Lucchi++/Train_In/"
# search text files starting with the word "sales"
pattern = path + "*.png"

# List of the files that match the pattern
result = glob.glob(pattern)

# Iterating the list with the count
count = 0
for file_name in result:
    old_name = file_name
    new_name = path + str(count) + ".png"
    os.rename(old_name, new_name)
    count = count + 1

# printing all revenue txt files
res = glob.glob(path  + "*.png")
for name in res:
    print(name)