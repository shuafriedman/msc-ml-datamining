#!/usr/bin/python3
import sys
 
# maps words to their counts
current_word = None
current_count = 0 
for line in sys.stdin:
    w = line.split()[0] # this is the word
    if w != current_word and w != None:
        print('%s\t%s' % (current_word, current_count))
        current_word = w
        current_count = 0
    current_count+=1

# Write (unsorted) tuples to stdout
    
