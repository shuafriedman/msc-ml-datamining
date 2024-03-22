# SAM-for-Traffic-Classification

Paper title: Self-attentive deep learning method for online traffic classification and its
interpretability

**Run the files as follow:**

STEPS TO RUN
1. Downlaod the dataset-- The dataset is available at http://mawi.wide.ad.jp/mawi/samplepoint-G/2020/202006101400.html
2. run the preprocess.py file on the resulting pcap file
3. run the tools.py file on the resulting pkl file (splits into train and test, as well as other pre-processing)
4. run the train.py file.

# Data Structure
Each packet, after running tools.py, contains 3 elements: x = header_data, y = position in packet, label = protocol type (integer 0-9)
X and Y are both fed into the model as input, predicting the label.