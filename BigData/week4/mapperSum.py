#!/usr/bin/python3

import sys
import string

for line in sys.stdin:
	if line.isspace()==False:
		line = line.strip()
		product,count = line.split("\t")
		print('%s\t%s' % (product, count))

