#!/usr/bin/python3
import sys
import string
 
#--- get all lines from stdin ---
for line in sys.stdin:
    #--- split the line into words ---
    translator = str.maketrans('', '', string.punctuation)
    words = line.lower().translate(translator).split()

    #--- output tuples [word, 1] in tab-delimited format---
    for word in words: 
        print('%s\t%s' % (word, "1"))
