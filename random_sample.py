from pathlib import Path
# import os
import random
import shutil

pathlist = Path("train/up/").glob('**/*.jpg')
nof_samples = 50 # down: 50 ; left: 115 ; right: 115 ; stop: 32 ; up: 50 

# tmp = []
testfiles = []
for k, path in enumerate(pathlist):
    if k < nof_samples:
        testfiles.append(str(path)) # because path is object not string
    else:
        i = random.randint(0, k)
        if i < nof_samples:
            testfiles[i] = str(path)
# print(len(testfiles))
# print(testfiles)
# testfiles.append(tmp)

for file_name in testfiles:
    shutil.copy(file_name, 'test/')