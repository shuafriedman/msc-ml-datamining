# -*- coding: utf-8 -*-
# @Author: xiegr
# @Date:   2020-09-06 18:07:51
# @Last Modified by:   xiegr
# @Last Modified time: 2021-06-03 17:16:25
import torch
import os
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import argparse
import time
import pandas as pd
from tqdm import tqdm, trange
from ShuaImprovedSam import SAM
from tool import protocols, load_epoch_data, max_byte_len
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score, classification_report
import pickle
import numpy as np
from pathlib import Path
#set a random seed
seed = 0
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Dataset(torch.utils.data.Dataset):
	"""docstring for Dataset"""
	def __init__(self, x, y, label):
		super(Dataset, self).__init__()
		self.x = x
		self.y = y
		self.label = label

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		return self.x[idx], self.y[idx], self.label[idx]

def paired_collate_fn(insts):
	x, y, label = [np.array(x) for x in list(zip(*insts))]
	return torch.LongTensor(x), torch.LongTensor(y), torch.LongTensor(label)

def cal_loss(pred, gold, cls_ratio=None):
	gold = gold.contiguous().view(-1)
	# By default, the losses are averaged over each loss element in the batch. 
	loss = F.cross_entropy(pred, gold)

	# torch.max(a,0) 返回每一列中最大值的那个元素，且返回索引
	pred = F.softmax(pred, dim = -1).max(1)[1]
	# 相等位置输出1，否则0
	n_correct = pred.eq(gold)
	acc = n_correct.sum().item() / n_correct.shape[0]

	return loss, acc*100

def test_epoch(model, test_data):
	''' Epoch operation in training phase'''
	model.eval()

	total_acc = []
	total_pred = []
	total_score = []
	total_time = []
	# tqdm: 进度条库
	# desc ：进度条的描述
	# leave：把进度条的最终形态保留下来 bool
	# mininterval：最小进度更新间隔，以秒为单位
	for batch in tqdm(
		test_data, mininterval=2,
		desc='  - (Testing)   ', leave=False):

		# prepare data
		src_seq, src_seq2, gold = batch
		src_seq, src_seq2, gold = src_seq.to(device), src_seq2.to(device), gold.to(device)
		gold = gold.contiguous().view(-1)

		# forward
		if torch.cuda.is_available():
			torch.cuda.synchronize()
		start = time.time()
		pred, score = model(src_seq, src_seq2)
		if torch.cuda.is_available():
			torch.cuda.synchronize()
		end = time.time()
		# 相等位置输出1，否则0
		n_correct = pred.eq(gold)
		acc = n_correct.sum().item()*100 / n_correct.shape[0]
		total_acc.append(acc)
		total_pred.extend(pred.long().tolist())
		total_score.append(torch.mean(score, dim=0).tolist())
		total_time.append(end - start)

	return sum(total_acc)/len(total_acc), np.array(total_score).mean(axis=0), \
	total_pred, sum(total_time)/len(total_time)

def train_epoch(model, training_data, optimizer):
	''' Epoch operation in training phase'''
	model.train()

	total_loss = []
	total_acc = []
	# tqdm: 进度条库
	# desc ：进度条的描述
	# leave：把进度条的最终形态保留下来 bool
	# mininterval：最小进度更新间隔，以秒为单位
	for batch in tqdm(
		training_data, mininterval=2,
		desc='  - (Training)   ', leave=False):

		# prepare data
		src_seq, src_seq2, gold = batch
		src_seq, src_seq2, gold = src_seq.to(device), src_seq2.to(device), gold.to(device)

		optimizer.zero_grad()
		# forward
		pred = model(src_seq, src_seq2)
		loss_per_batch, acc_per_batch = cal_loss(pred, gold)
		# update parameters
		loss_per_batch.backward()
		optimizer.step()

		# 只有一个元素，可以用item取而不管维度
		total_loss.append(loss_per_batch.item())
		total_acc.append(acc_per_batch)

	return sum(total_loss)/len(total_loss), sum(total_acc)/len(total_acc)

def main(i, flow_dict):
    # if results file doesn't exist, open it
    
	filename = 'results/results_%d.txt'%i
	dirname = os.path.dirname(filename)

	# If the directory doesn't exist, create it
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	# Check if the file doesn't exist
	if not os.path.exists(filename):
		# Open the file in write mode, which will create the file
		with open(filename, 'w') as f:
			f.write('Train Loss Time Test\n')
			f.flush()

	model = SAM(num_class=len(protocols), max_byte_len=max_byte_len).to(device)
	optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()))
	loss_list = []
	# default epoch is 3
	for epoch_i in trange(5, mininterval=2, \
		desc='  - (Training Epochs)   ', leave=False):

		train_x, train_y, train_label = load_epoch_data(flow_dict, 'train')
		dataset = Dataset(x=train_x, y=train_y, label=train_label)
		#get only subset of the dataset
		# dataset = torch.utils.data.Subset(dataset, range(1000))
		num_samples = int(1.0 * len(dataset))

		# Generate random indices without replacement
		indices = np.random.choice(len(dataset), num_samples, replace=False)

		# Create the subset
		dataset = torch.utils.data.Subset(dataset, indices)
		training_data = torch.utils.data.DataLoader(
				dataset,
				num_workers=0,
				collate_fn=paired_collate_fn,
				batch_size=128,
				shuffle=True
			)
		train_loss, train_acc = train_epoch(model, training_data, optimizer)

		test_x, test_y, test_label = load_epoch_data(flow_dict, 'test')

		test_data = torch.utils.data.DataLoader(
				Dataset(x=test_x, y=test_y, label=test_label),
				num_workers=0,
				collate_fn=paired_collate_fn,
				batch_size=128,
				shuffle=False
			)
		test_acc, score, pred, test_time = test_epoch(model, test_data)


		# write ACC
		accuracy = accuracy_score(test_label, pred)
		precision, recall, fscore, _ = precision_recall_fscore_support(test_label, pred, average='macro')
		# Generate a confusion matrix
		conf_matrix = confusion_matrix(test_label, pred)

		# Generate a classification report with label names
		report = classification_report(test_label, pred, target_names=protocols)

		# Optional: Convert confusion matrix to a DataFrame for better readability
		conf_matrix_df = pd.DataFrame(conf_matrix, index=protocols, columns=protocols)

		# Write results to a text file
		# with open('evaluation_results.txt', 'w') as f:
		# 	f.write(f"Accuracy: {accuracy}\n\n")
		# 	f.write("Confusion Matrix:\n")
		# 	conf_matrix_df.to_csv(f, sep='\t')  # Writing the DataFrame to a file for better readability
		# 	f.write("\nClassification Report:\n")
		# 	f.write(report)

		print("Evaluation results with class labels written to evaluation_results.txt")
		# # early stop
		# if len(loss_list) == 5:
		# 	if abs(sum(loss_list)/len(loss_list) - train_loss) < 0.005:
		# 		break
		# 	loss_list[epoch_i%len(loss_list)] = train_loss
		# else:
		# 	loss_list.append(train_loss)

		# f.close()
		# return accuracy, p, r, fscore, report
		return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': fscore}


if __name__ == '__main__':
	print("Current working directory: ", os.getcwd())
	scores_dict = {
		'accuracy': [],
		'precision': [],
		'recall': [],
		'f1': [],
	}
	for i in range(0,10):
		path = Path(__file__).parent / f'data/pro_flows_{i}_noip_fold.pkl'
		with path.open('rb') as f:
			flow_dict = pickle.load(f)
		print('====', i, ' fold validation ====')
		scores = main(i, flow_dict)
		for key, value in scores.items():
			scores_dict[key].append(value)
   # average the scores
	for key, value in scores_dict.items():
		scores_dict[key] = sum(value) / len(value)
	with open('scores_improvedSam2.txt', 'w') as f:
		for key, value in scores_dict.items():
			f.write(f'{key}: {value}\n')