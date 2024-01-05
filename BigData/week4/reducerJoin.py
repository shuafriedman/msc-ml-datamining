#!/usr/bin/python3
import sys

# maps words to their counts
current_channel = None
current_count = 0
for line in sys.stdin:
    channel, product, count = line.split('\t')
    if product == '-':
        current_count = count
    else:
        print('%s\t%s' % (product, current_count))
# Write (unsorted) tuples to stdout
    
