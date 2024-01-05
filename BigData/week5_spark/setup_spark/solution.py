#!/usr/bin/env python
import pyspark
from pyspark import SparkContext
sc = SparkContext()

ads = sc.textFile("data/ads.txt").map(lambda x : x.split(',')).map(lambda x : [x[1], x[0]])
channels = sc.textFile("data/channels.txt").map(lambda x : x.split(','))
input = ads.join(channels).sortBy(lambda x: x[1][0])
output = input.map(lambda x: x[1]).reduceByKey(lambda a,b : int(a)+int(b)).sortBy(lambda x: x[1], ascending=False).collect()
print(output)