from os import listdir
from os.path import isfile, join
import glob

for name in glob.glob("labels/*.txt"):
    print(name)
    fd = open(name)
    count = 0
    for line in fd.readlines():
        print(line.strip())
        count = count + 1
    if count > 2:
        print("more than 2 classes error")
        break
    #print("The file contains", count, "lines.")
    fd.close()
