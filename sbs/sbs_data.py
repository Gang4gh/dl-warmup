import collections
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SBSPair = collections.namedtuple('SBSPair', 'query snippet1 snippet2 weight label')

def load_data(data_file='sbsdatacleaned20181008_flips_tokLower.tsv'):
	data = []
	prev = None
	with open(data_file) as f:
		for line in f:
			parts = line.rstrip().split('\t')
			if len(parts) != 7: continue
			if prev and prev.snippet1 == parts[2] and prev.snippet2 == parts[1]:
				continue
			data.append(SBSPair(*parts[0:3], int(parts[-2]), int(parts[-1])))
			prev = data[-1]
	print('load %d records from %s' % (len(data), data_file))

	ret = []
	random.seed(3)
	for sbs in data:
		if random.randint(0,99) % 2  == 1:
			ret.append(sbs._replace(snippet1=sbs.snippet2, snippet2=sbs.snippet1, label=-sbs.label))
		else:
			ret.append(sbs)

	return train_test_split(ret, test_size=0.1, random_state=1016)

def count_label_dist(data):
	label_dist = collections.Counter(sbs.label for sbs in data)
	print('label_dist(test_data):', label_dist)
	print('accuracy when always guess "0": %f' % (label_dist[0]/len(data)))

def calculate_accuracy_when_rank_by_length(training_data, test_data):
	for data, tag in zip([training_data, test_data], ['training', 'test']):
		pred = [np.sign(len(sbs.snippet1) - len(sbs.snippet2)) for sbs in data]
		y = [sbs.label for sbs in data]
		acc = accuracy_score(y, pred)
		print('accuracy(rank by length, include 0) on %s set is %f' % (tag, acc))
		pred = [np.sign(len(sbs.snippet1) - len(sbs.snippet2)) for sbs in data if sbs.label != 0]
		y = [sbs.label for sbs in data if sbs.label != 0]
		acc = accuracy_score(y, pred)
		print('accuracy(rank by length, ignore 0) on %s set is %f' % (tag, acc))

def find_best_accuracy_and_delta(data):
	# best accuracy/delta when assign '0' under 'abs(len(s1) - len(s2)) < delta'
	best_acc, best_delta = None, None
	y = [sbs.label for sbs in data]
	for delta in range(1, 200):
		pred = [0 if abs(len(sbs.snippet1) - len(sbs.snippet2)) < delta else np.sign(len(sbs.snippet1) - len(sbs.snippet2)) for sbs in data]
		acc = accuracy_score(y, pred)
		if best_acc is None or acc > best_acc:
			best_acc, best_delta = acc, delta
	print('best accuracy/delta is :', best_acc, best_delta)

if __name__ == "__main__":
	data_file_path = 'sbsdatacleaned20181008_flips_tokLower.tsv'
	training_data, test_data = load_data(data_file_path)
	val_offset = len(training_data) // 10 # split 10% training data as validation data
	random.Random(1016).shuffle(training_data)
	training_data = training_data[:-val_offset]
	with open('trainingdata.tsv', 'w') as f:
		for rec in training_data:
			f.write('{}\t{}\t{}\t{}\t{}\n'.format(rec.query, rec.snippet1, rec.snippet2, rec.weight, rec.label))
	count_label_dist(test_data)
	calculate_accuracy_when_rank_by_length(training_data, test_data)
	find_best_accuracy_and_delta(test_data)

#load 404908 records from sbsdatacleaned20181008_flips_tokLower.tsv
#label_dist(test_data): Counter({0: 22660, -1: 8974, 1: 8857})
#accuracy when always guess "0": 0.559631
#accuracy(rank by length, include 0) on training set is 0.373755
#accuracy(rank by length, ignore 0) on training set is 0.809598
#accuracy(rank by length, include 0) on test set is 0.376997
#accuracy(rank by length, ignore 0) on test set is 0.807975
#best accuracy/delta is : 0.5904275024079425 125

