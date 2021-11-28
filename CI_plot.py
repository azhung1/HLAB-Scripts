import pandas as pd
import argparse
from os import listdir
import sys
import numpy as np
import statistics


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Set params")
	parser.add_argument('--t', type = str, help = 'name of CV table')
	parser.add_argument('--pred_field', type = str, help = 'field with the predictions')
	parser.add_argument('--label_field', type = str, help ='field with original values')

	args = parser.parse_args()

df = pd.read_csv(args.t, usecols = ['Id', 'dep_score__withLanguage', 'dep_score_trues'])

t1_sorted = df.sort_values(by=['Id'])

print(t1_sorted.head(10))
print(df.sort_index())
squared_error = []
print(df.at[1, 'dep_score_trues'])

for index in range(0, df.shape[0]):
	squared_error.append((df.at[index, 'dep_score_trues']-df.at[index, 'dep_score__withLanguage']) **2)
overall_mse = statistics.mean(squared_error)
size = len(squared_error)
iters = 100000
resampled_means = []
for i in range(0, iters):
	resample = np.random.choice(squared_error, size = size, replace= True)
	resampled_means.append(statistics.mean(resample))
sorted_re_means = sorted(resampled_means)
print (size)

low95CI = sorted_re_means[round(0.025*iters)]
upper95CI = sorted_re_means[round(0.975*iters)]
