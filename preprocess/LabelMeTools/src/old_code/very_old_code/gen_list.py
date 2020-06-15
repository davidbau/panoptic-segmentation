import os
import sys
import glob

root = sys.argv[1]
depth = int(sys.argv[2])

folders = "".join(["*/" for i in xrange(depth)])
pattern = folders + "*.jpg"
print pattern

list_im = glob.glob(os.path.join(root, pattern))
list_im.sort()

print len(list_im)
with open('images.txt','w') as f:
    num = len(pattern.split('/'))
    for line in list_im:
        split = line.split('/')

        line = '/'.join(split[len(split)-num:])
        f.write(line + '\n')
