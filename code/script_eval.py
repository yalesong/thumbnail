import os
import csv
import glob
import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


labels_dir = os.path.join('..', 'labels')
results_dir = os.path.join('..', 'results')
dataset_tsv = os.path.join('..', 'videos', 'yahoo_thumbnail_cikm2016.tsv')
export_fig_path = '.'

# SIFTflow ditance threshold
theta_step = 0.0001
theta = np.arange(0, 0.005 + theta_step, theta_step)

# size of ranked list
top_k = 5

# Model names, to be used in results
model_names =['Random','K-means Centroid','K-means Stillness','G.-LASSO','Beauty Rank','CNN','Ours Supervised','Ours Unsupervised']
# Model directory names
model_dirs = ['random','kmeans_centroid','kmeans_stillness','glasso','beauty_rank','cnn','ours_supervised','ours_unsupervised']
# For table
latex_model_name = ['Random','K-means Centroid','K-means Stillness',\
	'G.-LASSO~\cite{cong2012towards}','Beauty Rank',\
	'CNN~\cite{yang2015blog}','Ours Supervised','Ours Unsupervised']
pred_dirs = [os.path.join(results_dir, model_dir) for model_dir in model_dirs]

# Categories -> figure legend mapping
xtick_name = ['Auto','Celeb','Comedy','Cute','Fashion','Finance','Food',\
	'Game','Health','Intl.','Makers','Movie','News','Parent','Sports',\
	'TV','Tech.','Travel','','All']

models = pd.DataFrame(data = {'model_name': model_names, 'pred_dir': pred_dirs})

# Get list of video IDs
videos_label_paths = glob.glob(os.path.join(labels_dir, '*.txt'))
videos = [os.path.splitext(os.path.basename(x))[0] for x in videos_label_paths]

# Columns in same order as TSV
tsv_columns = ['video_id', 'category', 'title', 'page_url', 'video_url']
columns = tsv_columns + ['label_path']
data = pd.DataFrame(data = { 'video_id': videos, 'label_path': videos_label_paths }, index=videos, columns=columns)

# Get the video metadata
with open(dataset_tsv, 'r', encoding='utf-8') as fin:
	reader = csv.reader(fin, delimiter='\t')
	for row in reader:
		target_row = data.loc[row[0]]
		for i, col in enumerate(tsv_columns):
			if i != 0:
				target_row[col] = row[i]

## Evaluate Precision@k
precision_k = np.zeros((len(videos), len(models), len(theta), top_k))
for video_idx, (video_id, video) in enumerate(data.iterrows()):
	if video_idx % 100 == 0:
		print(f'Reading results {video_idx}/{len(videos)}\n')
	
	with open(video.label_path, 'r') as fin:
		reader = csv.reader(fin)
		labels = np.array([float(row[0]) for row in reader])
	
	for model_idx, model in models.iterrows():
		pred_path = os.path.join(model.pred_dir, f'{video_id}.txt')
		with open(pred_path, 'r') as fin:
			reader = csv.reader(fin)
			# Python indexes from 0, Matlab indexes from 1
			pred = [int(row[0]) - 1 for row in reader]
			# MATLAB sometimes ignores the last frame of video; we cap the max frame index
			# We keep this behavior to keep the results consistent with the MATLAB implementation
			pred = [max(0, min(x, len(labels) - 1)) for x in pred]
			for theta_idx, theta_threshold in enumerate(theta):
				labels_mask = labels <= theta_threshold
				matches = labels_mask[pred]
				for top_k_idx in range(top_k):
					relevant_matches = matches[:min(top_k_idx+1, len(pred))]
					precision_k[video_idx, model_idx, theta_idx, top_k_idx] = sum(relevant_matches) > 0

# Perform significance test
significance = np.zeros((len(models), len(models), top_k))
# Max theta
target_theta_idx = len(theta) - 1
for i, model in models.iterrows():
	for j in range(i+1, len(models)):
		for k in range(top_k):
			ri = precision_k[:, i, target_theta_idx, k]
			rj = precision_k[:, j, target_theta_idx, k]
			_, significance[i,j,k] = scipy.stats.ttest_ind(ri, rj)
			significance[j,i,k] = significance[i,j,k]

## Print out results
for i, model in models.iterrows():
	print('{0:30}:\t'.format(model.model_name), end='')
	for top_k_idx in range(top_k):
		print('{0:.4f}\t'.format(np.mean(precision_k[:,i,-1,top_k_idx])), end='')
	print('')

print_significance_test_results = True
if print_significance_test_results:
	significance_test_bar = 0.05
	reference_model_index = 7
	print(f'\nStatistical significance (reference: {models.loc[reference_model_index].name}, index: {reference_model_index})')
	for i, model in models.iterrows():
		if i == reference_model_index:
			continue
		print('{0:30}:\t'.format(model.model_name), end='')
		for j in [1, 3, 5]:
			pvalue = significance[reference_model_index,i,j-1]
			if pvalue <= significance_test_bar:
				print(f'(k={j}) p={pvalue}*\t', end='')
			else:
				print(f'(k={j}) p={pvalue}\t', end='')
			if j==5:
				print('')


## Detailed results for paper
generate_paper_results = True
save_figure = False
if generate_paper_results:
	# 
	# Table 1: Aggregated performance score. Set P@k=1,3,5,10, theta=0.005,
	#
	PatK = np.zeros((len(models),top_k))
	for i in range(len(models)):
		for j in range(top_k):
			PatK[i,j] = np.mean(precision_k[:, i, target_theta_idx, j])

	# significance wrt 'Ours Unsupervised'
	reference_model_index = 7
	sort_idx = np.argsort(-PatK, axis=0)
	for i in range(len(models)):
		print(f'\t\t{latex_model_name[i]} & ', end='')
		for j in [0, 2, 4]:
			pvalue = significance[reference_model_index, i, j]
			mark=''
			if pvalue < 0.001:
				mark = '$^{\ddagger}$'
			elif pvalue < 0.05:
				mark = '$^{\dagger}$'

			if i == sort_idx[0,j]:
				print('\\textbf{{{0:.4f}}}% '.format(PatK[i,j]), end='')
			elif i == sort_idx[1,j]:
				print('\\underline{{{0:.4f}{1}}} '.format(PatK[i,j], mark), end='')
			else:
				print('{0:.4f}{1} '.format(PatK[i,j], mark), end='')

			if j == 4:
				if i == 5:
					print('\\\\ \\hline')
				else:
					print('\\\\ ')
			else:
				print('& ', end='')

	# 
	# Figure. Full spectrum of P@k
	# 
	fig1, ax1 = plt.subplots()
	r = np.transpose(np.mean(precision_k[:,:,:,top_k-1], axis=0))
	ax1_linestyle = ['-.','--','--','-.',':','-.','-','-']
	ax1_linecolor = [[4, 15, 141], [10, 52, 245], [34, 175, 247], [78, 254, 192],\
					[0, 0, 0], [253, 158, 40], [252, 23, 26], [126, 3, 8]]
	ax1_linecolor = np.array(ax1_linecolor) / 255.
	ax1_linewidth = np.ones(len(models))*2
	
	ax1_lines = ax1.plot(theta, r)
	for i, line in enumerate(ax1_lines):
		line.set_linestyle(ax1_linestyle[i])
		line.set_color(ax1_linecolor[i])
		line.set_linewidth(ax1_linewidth[i])
	
	ax1.legend(model_names, loc='upper left')
	ax1.set_title('Precision@K (K=5)')
	ax1.set_ylabel('Mean Precision@k')
	ax1.set_xlabel('SIFTflow dist threshold ($\\theta$)')
	ax1.set_xlim(theta[0], theta[-1])
	plt.get_current_fig_manager().window.setGeometry(800, 500, 740, 680)
	
	if save_figure:
		plt.savefig(os.path.join(export_fig_path, 'prec_k_cikm2016.eps'), transparent=True, format='eps')

	
	# 
	# Figure. Bar plot per channel. Set P@5, theta=0.005
	#
	categories_unique = np.unique(data.category.values)
	p_k_per_category = np.zeros((len(categories_unique),len(models)))
	for i, category in enumerate(categories_unique):
		video_mask = data.category.values == category
		p_k_per_category[i,:] = np.mean(precision_k[video_mask, :, target_theta_idx, top_k-1], axis=0)

	p_k_per_category = np.vstack([p_k_per_category, np.zeros((1, len(models))),\
						np.mean(precision_k[:, :, target_theta_idx, top_k-1], axis=0)])
	
	p_k_per_category_dataframe = pd.DataFrame(p_k_per_category, index=xtick_name, columns=model_names)
	ax2 = p_k_per_category_dataframe.plot.bar(colormap='jet', rot=90, width=0.75)
	
	ax2.set_xticklabels(p_k_per_category_dataframe.index, rotation=0)

	plt.get_current_fig_manager().window.setGeometry(1, 549, 1680, 606)
	ax2.set_xlim(-1, len(xtick_name))
	ax2.set_ylim(0, 0.6)
	ax2.set_title('Precision@K per channel (K=5, $\\theta$=0.005)')

	if save_figure:
		plt.savefig(os.path.join(export_fig_path, 'prec_k_channel_cikm2016.eps'), transparent=True, format='eps')

