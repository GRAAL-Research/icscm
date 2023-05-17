import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_results_list = []
#list files in directory:
results_dir = 'cardinality-exp-results'

for file in os.listdir(results_dir):
    df_loc = pd.read_csv(os.path.join(results_dir, file))
    df_results_list.append(df_loc)
big_perf_df = pd.concat(df_results_list)

n_samples_of_plot = 10000
big_perf_df = big_perf_df[big_perf_df['n_samples']==n_samples_of_plot]

big_perf_df['algo'].replace({'ICP+DT': 'ICP'}, inplace=True)
algos_to_keep = ['SCM', 'DT', 'ICP', 'ICSCM']

big_perf_df = big_perf_df[big_perf_df['algo'].isin(algos_to_keep)]
perf_df_runtime = big_perf_df[big_perf_df['type'] == 'fit time']
#figure size:
plt.figure(figsize=(6, 3))
plt.rcParams.update({'font.size': 15})
sns.lineplot(data=perf_df_runtime, x='n_var', y='score', hue='algo', hue_order=algos_to_keep, markers=True, dashes=False)
plt.ylabel('runtime (seconds)')
plt.xlabel('$|X_b|$')
plt.yscale('log')

# Put a legend to the top of the figure:
plt.legend(loc='center', bbox_to_anchor=(1.16, 0.5), ncol=1, fancybox=True, shadow=False)
plt.xticks(sorted(list(set(big_perf_df['n_var']))))
a = []
for b in np.array([1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02]):
    a.extend(np.arange(1*b, 10*b, b))
a.append(1000)
plt.yticks(a)
# save figure with bbox_inches='tight' to avoid cropping the legend
plt.savefig(f'runtime_plot_nsamples_{n_samples_of_plot}.pdf', bbox_inches='tight')
plt.show()

perf_df_causalscore = big_perf_df[big_perf_df['type'] == 'causal']

min_n_splits = 1000
for a in set(perf_df_causalscore['algo']):
    perf_df_causalscore_a = perf_df_causalscore[perf_df_causalscore['algo'] == a]
    for n in set(perf_df_causalscore['n_var']):
        perf_df_causalscore_a_n = perf_df_causalscore_a[perf_df_causalscore_a['n_var'] == n]
        splits = set(perf_df_causalscore_a_n['split'])
        #print(a, n, len(splits))
        if 0 < len(splits) < min_n_splits:
            min_n_splits = len(splits)
print(min_n_splits)
perf_df_causalscore = perf_df_causalscore[perf_df_causalscore['split'].isin(list(range(min_n_splits)))]

#ax_id += 1
#ax[ax_id].set_ylabel('causal score')
## heatmap
plt.figure(figsize=(5, 3))
perf_df_causalscore['cardinality of Xb'] = perf_df_causalscore['n_var']
del perf_df_causalscore['n_var']
perf_df_causalscore_small = perf_df_causalscore[['algo', 'score', 'split', 'cardinality of Xb']]
heatmap_df = perf_df_causalscore_small.groupby(['algo', 'cardinality of Xb']).mean().reset_index().pivot(index='algo', columns='cardinality of Xb', values='score')
heatmap_df = heatmap_df.reindex(algos_to_keep)
print(heatmap_df.index)

sns.heatmap(heatmap_df, annot=True, cmap='Greens')
plt.title('proportion of identification of causal parents over {} splits'.format(min_n_splits), y=1.00)
plt.show()