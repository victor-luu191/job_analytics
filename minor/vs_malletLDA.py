import cluster_skill_helpers as cluster_skill_helpers
from cluster_skill_helpers import *

MALLET_LDA = 'd:/larc_projects/job_analytics/results/skill_cluster/new/mallet_lda/'
k=30; dir_name = MALLET_LDA + '{}topics/'.format(k)

topics = pd.read_csv(dir_name + 'topics.csv')['topic']

topic_assign = pd.read_csv(dir_name + 'job_analytic_compose.tsv', sep='\t', header=None)
# convert topic assignments to matrix form
mallet_doc_topic = np.asarray(topic_assign.iloc[:, 2:])
# del getTopTopicProb; del getTopTopics4Docs; del getTopTopics_GT
print getTopTopics(mallet_doc_topic[13, :], topics)