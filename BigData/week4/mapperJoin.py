#!/usr/bin/python3
import sys
import string
 
#--- get all lines from stdin ---
for line in sys.stdin:
    #initialize
    product = '-'
    channel = '-'
    count = 0
    a, b = [x.strip() for x in line.split(',')]
    try:
        count = int(b)
        channel = a
    except:
        product = a
        channel = b
    product = product.replace(' ', '')

    print('%s\t%s\t%s' % (channel, product, count))
