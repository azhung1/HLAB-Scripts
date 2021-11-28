import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import random
import sys

if __name__ == '__main__':

	print('Starting...')
	sm_robert = np.array([.7497, .7486, .7422, .7396, .7369, .7364, .7365, .735, .7384, .7359, .7356, .7368])
	temp = np.empty(12)
	temp[:] = np.nan
	sm_robert = np.append(sm_robert, temp) # pad to 24
	lg_robert = np.array([.7482, .7479, .7507, .7508, .748, .7403, .7391, .7363, .7365, .7377, .7364, .7352, .7323, .7304, .7305, .7284, .7272, .7264, .7257, .7275, .7265, .7263, .7282, .7293])

	sm_roberta_se = np.array([.0068, .0071, .007, .0068, .0064, .0063, .0066, .0066, .0064, .0064, .0064, .006])
	sm_roberta_se = np.append(sm_roberta_se, temp)
	lg_roberta_se = np.array([.0061, .0061, .0063, .0065, .0066, .0061, .0059, .0057, .0058, .0057, .006, .0058 , .0061, .0067, .0065, .0063, .006, .006, .006, .0062, .0061, .0059, .0063, .0063])
	print(len(lg_roberta_se))

	df = pd.DataFrame({'RoBERTa-large': lg_robert, 'layers': [i for i in range(1,25)]})
	df = pd.melt(df, ['layers'])
    
	df.columns = ['Layers', 'model', 'MSE']
	#print(df.head())
	errors = [.01 for i in range(len(lg_robert))]
	df['se'] = errors
	# print(df.head())

	# configure plot params
	params = {'axes.facecolor':'white', 'figure.figsize':(22,14), 'font.size':32, 'legend.fontsize':32, 'xtick.labelsize':20, 'ytick.labelsize':20,
    	'axes.labelsize':32, 'legend.title_fontsize': 32}
	# sns.set(rc=params)
	pal = sns.color_palette('pastel', 2)

	mpl.rcParams.update(params)
    
	x = [i for i in range(1,25)]
	fig, ax = plt.subplots(figsize=(22,14))
	ax.set_xticks(range(1,25))
	ax.set_xticklabels([i for i in range(1,25)])
	#ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
	plt.plot(x, lg_robert, color = 'black', linewidth = 10, label = 'RoBERTa-large', marker = 'o', markersize = 14, alpha = 0.5)
	plt.plot(x, sm_robert, color = 'red', linewidth = 10, label = 'RoBERTa-base', marker = 'o', markersize = 14, alpha = 0.5)
	plt.xlabel("Layers")
	plt.ylabel("MSE")
	plt.title("Single Layer Roberta")
	plt.legend(loc='best')
	plt.fill_between(x[:24], lg_robert[:24] - lg_roberta_se[:24], lg_robert[:24]+lg_roberta_se[:24], alpha = 0.5)
	plt.fill_between(x[:12], sm_robert[:12] - sm_roberta_se[:12], sm_robert[:12]+sm_roberta_se[:12], alpha = 0.5)
	ax.set_xlim(xmin = 1)
	ax.set_xlim(xmax = 24)
	plt.show()
	# plt.savefig('./single_layer_performance.png')


