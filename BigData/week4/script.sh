#!/usr/bin/bash
rm -rf mapper_tmp*
rm -rf output*
cat data/*.txt | local-mapreduce/lmr -k 300k 10 ./mapperJoin.py ./reducerJoin.py output 2>err
cat output/* > all_data.txt
cat all_data.txt | local-mapreduce/lmr -k 300k 10 ./mapperSum.py ./reducerSum.py output_final 2>err 
cat output_final/* | sort -grk 2 > final_output.txt
