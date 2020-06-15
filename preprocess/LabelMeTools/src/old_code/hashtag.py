import argparse
import os
import json
import numpy as np
import cv2

from changeRLE import maskToRLE

from pycocotools import mask as COCOmask

def rleToString(cnts):
  # p = 0
  s = ""
  for i, x in enumerate(cnts):
    if(i > 2):
      x -= cnts[i-2];
    more = 1;

    while ( more ):
      c = x & 0x1f
      x >>= 5
      if c & 0x10:
        more = x!=-1
      else:
        more = x!=0
      if(more):
        c |= 0x20;
      c += 48;
      s += chr(c)
  # s[p]=0;
  return s;

def rleFrString(s):
  m = 0
  p = 0
  cnts = []
  while p < len(s):
    x = 0
    k = 0
    more = 1
    while( more ):
      c = ord(s[p])-48
      x |= ((c & 0x1f) << 5*k)
      more = c & 0x20
      p += 1
      k += 1
      if not more and (c & 0x10):
        x |= -1 << 5*k
    if(m > 2):
      x += cnts[m-2]
    m += 1
    cnts.append(x)
  return cnts

def maskFrRLE(cnts, shape):
  b = False
  mask = []
  for c in cnts:
    for i in range(c):
      mask.append(b)
    b = not b
  while len(mask) < shape[0] * shape[1]:
    mask.append(0)

  mask = np.array(mask, dtype='uint8')
  mask = np.reshape(mask, shape[::-1])
  mask = np.transpose(mask)
  return mask


if __name__ == "__main__":
  size = [335,324]
  counts = "QcX1e0j9g0YO2N000000000O010000000000O10000000000O1000000000000000000O10000000000000000000000001O000000001O0000001O0000000000000000000000000000000000000000000000000000000000000O100000000000000000000O100000000000000000000"
  # size = [512,683]
  # counts = "bRm72n?1O1N3N1O2M2O1O2M2O1N3N1O1N3N1O1N3N1O2M2O1O2N1N2O2M2O1O2M2O2N1N2O2N1N2O1O2M2O2N1N2O2N1N3N1O1N2O2N1N2O2N1N3N1N2O2N1N3N1O1N2O2N1N3N1O1N3N1O2M2O1O1N3N1O1N3N1O2M2O1O2M2O1O2M2O1O2M2O1O2M2O1O0O1002N1O2N1O2N1N3N1O2N1O2N1O2N1N3N1O2N1O2N1O2N1O2N1O2M2O2N1O2N1O2N2N1O1O2N1N3N1O2N1O2N2N1O1O2N1O2N1N3N1O2N2N1O1O2N1O2N1O2M2O2N1O2N1O2N1O1N3N2N1O2N1O2N1O2N1O1O2M3Nh0XOo9"
  cnts = rleFrString(counts)
  # mask = maskFrRLE(cnts, size)
  # cv2.imshow("hi", mask * 255)
  # cv2.waitKey(0)
  s = rleToString(cnts)

  print("Orig:", counts)
  print("Mine:", s)



  rle = {}
  rle["size"] = size
  rle["counts"] = counts
  mask2 = COCOmask.decode(rle)
  encoded = COCOmask.encode(mask2)
  print(encoded)
  # print(np.array_equal(mask, mask2))



