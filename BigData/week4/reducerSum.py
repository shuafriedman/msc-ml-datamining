#!/usr/bin/python3
import sys
 
current_product = None
current_count = 0
for line in sys.stdin:
	if line:
		product, count = line.split('\t')
	if product != current_product:
		if current_product != None:       # if we aren't at the initialization step (first entry)
			print(current_product, current_count)
		current_product = product
		current_count = int(count)
	else:
		current_count+=int(count)
    
