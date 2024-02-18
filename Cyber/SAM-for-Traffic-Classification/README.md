# SAM-for-Traffic-Classification

Paper title: Self-attentive deep learning method for online traffic classification and its
interpretability

Accepted by Elsevier Computer Networks (https://doi.org/10.1016/j.comnet.2021.108267)  

More information about us https://xgr19.github.io

NetAI20 version https://github.com/xgr19/SAM-for-Traffic-Classification/tree/SAM-before-NetAI

**Run the files as follow:**

1. python3 preprocess.py
2. python3 tool.py
3. python3 train.py

The dataset is available at http://mawi.wide.ad.jp/mawi/samplepoint-G/2020/202006101400.html

STEPS TO RUN
1. Downlaod the dataset
2. run the preprocess.py file on the pcap file
3. run the tools.py file on the resulting pkl file (splits into train and test, as well as other pre-processing)
4. run the train.py file.

# Data Structure
Each packet, after running tools.py, contains 3 elements: x = header_data, y = position in packet, label = protocol type (integer 0-9)
X and Y are both fed into the model as input, predicting the label.

# Recomendations
1) More complex positional Encoder
2) More efficient Attention mechanism (sparse attention) # https://github.com/kyegomez/SparseAttention?tab=readme-ov-file

(rest)
3) layer normalization/batch normalization
4) Expand max byte len to 60 (google if this makes sense, instead of 50)