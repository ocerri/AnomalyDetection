import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("output", type=str, help='Name of output file')
parser.add_argument("inputs", type=str, help='Name of input files', nargs='+')
args = parser.parse_args()

list = []

for fpath in args.inputs:
    print fpath
    a = np.load(fpath)
    list.append(a)
    print a.shape

print 'Saving outpout in:', args.output
out = np.concatenate((tuple(list)))
print out.shape
np.save(args.output, out)
