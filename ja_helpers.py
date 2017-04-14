import random
import sklearn.feature_extraction.text as text_manip
import matplotlib.ticker as mtick

from sklearn.decomposition import NMF, LatentDirichletAllocation
from scipy.sparse import *
from scipy.io import *
from collections import Counter
from time import time

# my own modules
import my_util as my_util
from my_util import *
from stat_helpers import *

## Seed for reproducibility
# random.seed(123)

## Helpers for filtering
max_n_word = 3
def filterJDs(post_df, skills, min_n_skill=2):
    print('Extracting JDs with at least %d unique skills...' %min_n_skill)
    n_post, n_skill = post_df.shape[0], len(skills)
    # Count no. of unique skills in each JD
    binary_vectorizer = text_manip.CountVectorizer(vocabulary=skills, ngram_range=(1, max_n_word), binary=True)
    t0 = time()
    print('\tMarking occurrence of %d skills in %d JDs...' %(n_skill, n_post))
    doc_skill_occurrence = binary_vectorizer.fit_transform(post_df['clean_text'])
    print('Done after %.1fs' %(time() - t0))

    post_df['n_uniq_skill'] = doc_skill_occurrence.sum(axis=1).A1

    ## Remove JDs with <= 1 skills
    cond = 'n_uniq_skill >= {}'.format(min_n_skill)
    sub_posts = post_df.query(cond)
    return sub_posts

def filterSkills(skills, posts, min_n_jd):
    print('Extracting skills occuring in at least %d JDs...' %min_n_jd)
    # (Re-)count no. of JDs containing each skill
    n_posts = posts.shape[0]
    
    t0 = time()
    print('\tMarking occurrence of skills in %d JDs ...' %n_posts)
    binary_vectorizer = text_manip.CountVectorizer(vocabulary=skills, ngram_range=(1, max_n_word), binary=True)
    doc_skill_occurrence = binary_vectorizer.fit_transform(posts['clean_text'])
    print('Done after %.1fs' %(time() - t0))

    n_jd_by_skill = doc_skill_occurrence.sum(axis=0).A1
#     print quantile(n_jd_by_skill)

    # Remove skills occuring in  <=1  JDs
    df = pd.DataFrame({'skill': skills, 'n_jd_with_skill': n_jd_by_skill})
    cond = 'n_jd_with_skill >= {}'.format(min_n_jd)
    sub_skill_df = df.query(cond)
    return sub_skill_df

def cal_test_err(mf_models): 
    test_error = []
    print('No. of topics, Test error, Running time')

    for k in ks:
        t0 = time()
        H = mf_models[k].components_
        W_test = mf_models[k].fit_transform(X_test, H=H)
        err = la.norm(X_test - np.matmul(W_test, H))
        
        test_error.append(err)
        print('%d, %0.1f, %0.1fs' %(k, err, time() - t0))
#     end for
    return test_error

def findOccurSkills(init_skills, jd_docs):
    count_vectorizer = text_manip.CountVectorizer(vocabulary=init_skills, ngram_range=(1, max_n_word))
    t0 = time()
    print('Counting occurrence of skills with length <= %d ...' %max_n_word)
    doc_skill_freq = count_vectorizer.fit_transform(jd_docs)
    print('Done after %.1fs' %(time() - t0))

    skill_freq = doc_skill_freq.sum(axis=0).A1
    skill_df = pd.DataFrame({'skill': init_skills, 'total_freq': skill_freq})
    occur_skills_df = skill_df.query('total_freq > 0')
    occur_skills = occur_skills_df['skill']
    print('No. of skills actually occurring in JDs: %d' %len(occur_skills))
    return occur_skills_df

# def findSkills(occur_skills, jd_docs):
#     count_vectorizer = text_manip.CountVectorizer(vocabulary=occur_skills, ngram_range=(1, max_n_word))
#     t0 = time()
#     print('Counting occurrence of skills with length <= %d ...' %max_n_word)
#     doc_skill_freq = count_vectorizer.fit_transform(jd_docs)

#     print('Doing inverse transform to get skills in each JD...')
#     skills_by_jd = count_vectorizer.inverse_transform(doc_skill_freq)
#     print('Done after %.1fs' %(time() - t0))
#     return skills_by_jd

def filtering(init_posts, init_skills):
    n_iter, posts, skills = 0, init_posts, init_skills
    n_post = posts.shape[0]

    stop_cond, thres = False, .98
    while not stop_cond:
        n_iter = n_iter + 1
        print('Iteration %d' %n_iter)
        new_posts = extractJDs(posts, skills, min_n_skill=2)
        n_new_post = new_posts.shape[0]
        print('No. of posts after filtering: %d' %n_new_post)
        
        skill_df = extractSkills(skills, new_posts, min_n_jd=2)
        new_skills = skill_df['skill']
        print('No. of skills after filtering: %d' %len(new_skills) )
        stop_cond = (n_new_post >= thres*n_post) and (len(new_skills) >= thres*len(skills))
        
        posts = new_posts
        n_post = posts.shape[0]
        skills = new_skills
    # end while
    return posts, skills

def countOccur_ngram(n=1):
    t0 = time()
    print('Marking occurrence of {}-gram skills...'.format(n))
    # use option binary to indicate that we only care whether a given skill occurs or not, not the freq of the skill
    vectorizer = text_manip.CountVectorizer(vocabulary=skills, binary=True, ngram_range=(n,n)) 
    doc_ngram_occurrence = vectorizer.fit_transform(jd_docs)
    print('Done after %.1fs' %(time() - t0))
    n_ngram_by_jd = doc_ngram_occurrence.sum(axis=1).A1
    return pd.DataFrame({'job_id': posts['job_id'], 'n_{}gram'.format(n): n_ngram_by_jd})

def buildDocNgramMat(n, jd_docs, skills):
    t0 = time()
    print('Counting occurrence of {}-gram skills...'.format(n))
    vectorizer = text_manip.CountVectorizer(vocabulary=skills, ngram_range=(n,n))
    doc_ngram_mat = vectorizer.fit_transform(jd_docs)
    print('Done after %.1fs' %(time() - t0))
    return doc_ngram_mat

# global n_proc_doc
# n_proc_doc=0
def rmSkills(d, skills):
    ## there is problem with using a global count, turn off tmp
    global n_proc_doc
    n_proc_doc += 1; 
    if (n_proc_doc % 10000)==0:
        print('Removal for {} docs and counting...'.format(n_proc_doc))
    res = d
    for sk in skills:
        res = res.replace(sk, '')
    return res

def buildDocSkillMat(jd_docs, skill_df, folder):
    """
    @brief      {Build a document-skill matrix where each entry $e(d, s)$
                is the freq of skill $s$ in job description $d$. Handle 
                overlapping problem bw n-grams 
                (e.g. business in 2-gram 'business management' is regarded 
                    diff from business in 1-gram 'business')}
    
    @param      jd_docs   The clean jd documents
    @param      skill_df  The skill df
    @param      folder    The folder to store intermediate matrices 
                            {doc_unigram, doc_bigram, doc_trigram}, 
                            None if don't want to store them.
    
    @return     The sparse document-skill matrix.
    """
    
    def save(sp_mat, mat_name):
        fname = folder + mat_name + '.mtx'
        with(open(fname, 'w')) as f:
            mmwrite(f, sp_mat)
        print('Saved {} matrix'.format(mat_name))
        
    global n_proc_doc
    
    if not folder:
        print('No folder passed, will not save intermediate matrices.')

    trigram_skills = np.unique(skill_df.query('n_word == 3')['skill'])
    bigram_skills = np.unique(skill_df.query('n_word == 2')['skill'])
    unigram_skills = np.unique(skill_df.query('n_word == 1')['skill'])

    doc_trigram = buildDocNgramMat(n=3, jd_docs=jd_docs, skills=trigram_skills)
    if folder:
        save(doc_trigram, 'doc_trigram')
    print('Removing tri-grams from JDs to avoid duplications...')
    n_proc_doc = 0
    jd_docs = jd_docs.apply(rmSkills, skills=trigram_skills)
    print('Done')

    doc_bigram = buildDocNgramMat(n=2, jd_docs=jd_docs, skills=bigram_skills)
    if folder:
        save(doc_bigram, 'doc_bigram')
    print('Removing bi-grams from JDs...')
    n_proc_doc = 0
    jd_docs = jd_docs.apply(rmSkills, skills = bigram_skills)
    print('Done')

    doc_unigram = buildDocNgramMat(n=1, jd_docs=jd_docs, skills=unigram_skills)
    if folder:
        save(doc_unigram, 'doc_unigram')
    
    doc_skill = hstack([doc_unigram, doc_bigram, doc_trigram])
    return doc_skill

def getSkills(doc_idx, doc_term, skills):
    row = doc_term.getrow(doc_idx)
    indices = row.nonzero()[1]
    occur_skills = skills[indices]
    return pd.DataFrame({'occur_skills': ','.join(occur_skills), 'n_skill': len(occur_skills)}, index=[doc_idx])
    # including original document is meant for sanity checking
    # return pd.DataFrame({'doc': docs[doc_idx], 'occur_skills': ','.join(occur_skills), 'n_skill': len(occur_skills)}, index=[doc_idx])

def getSkills4Docs(docs, doc_term, skills): # doc_term is the doc-term count matrix built from docs (so train/test_docs go with train/test_doc_term resp)
    n_doc = len(docs)
    frames = [getSkills(doc_idx=dd, doc_term=doc_term, skills=skills) for dd in range(n_doc)]
    res = pd.concat(frames)
    # res = res.drop_duplicates()
    return res

def initLDA_model(k, beta):
    alpha = 50.0/k
    print("Init LDA with priors: alpha = %.1f, beta = %.1f" %(alpha, beta))
    model = LatentDirichletAllocation(n_topics=k, max_iter=5, learning_method='online', learning_offset=50., random_state=0, 
                                        doc_topic_prior=alpha, topic_word_prior=beta) # verbose=1
    return model

def trainLDA(beta, ks, trainning_set):
    
    lda_scores = []
    lda = {k: initLDA_model(k, beta) for k in ks}

    print('Fitting LDA models...')
    print('No. of topics, Log-likelihood, Running time')

    for k in ks:
        t0 = time()
        lda[k].fit(trainning_set)
        s = lda[k].score(trainning_set)
        print('%d, %0.1f, %0.1fs' %(k, s, time() - t0))
        lda_scores.append(s)
    
    return lda

def testLDA(lda, ks, test_set):
    
    perp = [lda[k].perplexity(test_set) for k in ks]
    perp_df = pd.DataFrame({'No. of topics': ks, 'Perplexity': perp})

    lda_best_k = ks[np.argmin(perp)]
    print('Best no. of topics for LDA: %d' %lda_best_k)
    return perp_df

def toIDF(terms, doc_term_mat):
    n_doc, n_term = doc_term_mat.shape[0], doc_term_mat.shape[1]
    
    # no. of docs containing a term = no. of non zero entries in the col of the term
    n_doc_with_term = [doc_term_mat.getcol(t).nnz for t in range(n_term)] 
    
    res = pd.DataFrame({'term': terms, 'n_doc_with_term': n_doc_with_term})
    res = res.query('n_doc_with_term > 0')
    res['idf'] = np.log10(np.divide(n_doc, n_doc_with_term))
    return res

def getClusterAtRow(i, df):
    r = df.iloc[i]
    cluster = r['cluster']
    prob = str(round(r['cluster_prob'], 3))
    s = ''.join([cluster, '(', prob, ')'])
    return s

# LDA_DIR = 'd:/larc_projects/job_analytics/results/skill_cluster/new/lda/'
# clusters = pd.read_csv(LDA_DIR + 'clusters.csv')['cluster']
# print('Loaded cluster labels as follow:')
# print(clusters)

def getTopClusterProb(row, doc_topic_distr):
    doc_idx = row.name
    probs = doc_topic_distr[doc_idx, :]
    return round(max(probs), 4)

def getTopClusters(k, doc_idx, doc_df, doc_topic_distr):
    probs = doc_topic_distr[doc_idx, :]
    df = pd.DataFrame({'cluster': clusters, 'cluster_prob': probs})
    df.sort_values('cluster_prob', ascending=False, inplace=True)
    
    top_k = [getClusterAtRow(i, df) for i in range(k)]

    row = doc_df.iloc[doc_idx]
    job_id, doc = row['job_id'], row['doc']
    return pd.DataFrame({'job_id': job_id, 'doc': doc, 'top_{}_cluster'.format(k): ';'.join(top_k)}, index=[doc_idx])

    # df.head(k)
    # df['job_id'] = doc_df.iloc[doc_idx]['job_id']
    # df['doc'] = doc_df.iloc[doc_idx]['doc']

def findIndex(arr, thres):
    sub_sums = {k: sum(arr[0:k]) for k in range(len(arr))}
    for k in range(len(arr)):
        if sub_sums[k] > thres:
            return k

def getTermsInDoc(row, doc_term_mat, vocab):
    idx = doc_term_mat[row].nonzero()[1]
    occur_terms = [vocab[i] for i in idx]
    return occur_terms

def getTopClusters_GT(row, doc_topic_distr, thres):
    doc_idx = row.name
    probs = doc_topic_distr[doc_idx, :]

    df = pd.DataFrame({'cluster': clusters, 'cluster_prob': probs})
    df.sort_values('cluster_prob', ascending=False, inplace=True)
    
    ## get top k clusters such that sum(prob_of_top_k) > thres
    k = findIndex(df['cluster_prob'], thres)
    top_k = [getClusterAtRow(i, df) for i in range(k)]
    return ';'.join(top_k)
    # return k

## Topics learned by MALLET LDA ====================================
def getTopicRow(i, df):
    row = df.iloc[i, :]
    topic, prob = row['topic'], str(round(row['prob'], 3))
    return ''.join([topic, '(', prob, ')'])

def getTopTopics(row, topics, thres=.5):
    df = pd.DataFrame({'topic': topics, 'prob': row})
    df = df.sort_values('prob', ascending=False)
    
    k = findIndex(df['prob'], thres)
    top_k = [getTopicRow(i, df) for i in range(k)]
    return ';'.join(top_k)

## Skill clustering analysis 
def plotSkillDist(res):
    fig = plt.figure()
    n, bins, patches = plt.hist(res['n_skill'], bins=np.unique(res['n_skill']))
    plt.xlabel('# skills in JD'); plt.ylabel('# JDs')
    plt.xticks(range(0, 120, 10))
    plt.grid(True)
    return fig

def getGroupMedian(g1, g2, g3, g4):
    m1 = np.median(g1['n_top_cluster']); m2 = np.median(g2['n_top_cluster'])
    m3 = np.median(g3['n_top_cluster']); m4 = np.median(g4['n_top_cluster'])

    print('Medians of the groups:')
    return pd.DataFrame({'range_of_n_skill': ['[2, 7)', '[7, 12)', '[12, 18)', '[18, 115]'], 
                            'median_of_n_top_cluster': [m1, m2, m3, m4]})

# x = [1,2,3,4]; labels = ['[2, 7)', '[7, 12)', '[12, 18)', '[18, 115]']
def mixtureSizePlot(g1, g2, g3, g4):
    groups = [g1['n_top_cluster'], g2['n_top_cluster'], g3['n_top_cluster'], g4['n_top_cluster']]
    
    fig = plt.boxplot(groups)
    plt.xlabel('# skills in job post'); plt.ylabel('Mixture size') # # most likely clusters
    
    plt.xticks(x, labels); plt.ylim(0, 9)
    return fig

def topClusterProbPlot(g1, g2, g3, g4):
    groups = [g1['prob_top_cluster'], g2['prob_top_cluster'], g3['prob_top_cluster'], g4['prob_top_cluster']]
    
    fig = plt.boxplot(groups)
    plt.xlabel('# skills in job post'); plt.ylabel('Probability of top cluster')
    plt.xticks(x, labels);
    plt.grid(True)

    return fig

def errorBarPlot(res, thres):
    g1 = res.query('n_skill < 7'); g2 = res.query('n_skill >= 7 & n_skill < 12')
    g3 = res.query('n_skill >= 12 & n_skill < 18'); g4 = res.query('n_skill >= 18')
    print('# posts in 4 groups:'); print(','.join([str(g1.shape[0]), str(g2.shape[0]), str(g3.shape[0]), str(g4.shape[0])]))

    ## Cal avg, min, max of each group
    col = 'n_top_cluster_{}'.format(int(thres*100))
    groups = [g1[col], g2[col], g3[col], g4[col]]
    min_groups = np.asarray(map(min, groups)); max_groups = np.asarray(map(max, groups)); 
    avg_groups = np.asarray(map(np.mean, groups))

    ## Plot
    lower_error = avg_groups - min_groups; upper_error = max_groups - avg_groups
    asymmetric_error = [lower_error, upper_error]

    y = avg_groups
    fig = plt.errorbar(x, y, yerr=asymmetric_error, fmt='o')
    
    plt.xlim(0, 5); 
    plt.ylim(0, 7)
    plt.xticks(x, labels); plt.grid(True)
    plt.xlabel('# skills in job post'); plt.ylabel('# skill clusters assigned to job post')
    plt.title('Mixture size (threshold={})'.format(thres))
    return fig

def getIndex(i, df):
    return df.iloc[i].name

## Topic modelling ====================================
# get top words of a topic (i.e. a word dist)
def get_top_words(n_top_words, word_dist, feature_names):
    norm_word_dist = np.divide(word_dist, sum(word_dist))
    sorting_idx = word_dist.argsort()
    top_words = [feature_names[i] for i in sorting_idx[:-n_top_words - 1:-1]]
    probs = [norm_word_dist[i] for i in sorting_idx[:-n_top_words - 1:-1]]
    
    return pd.DataFrame({'top_words': top_words, 'word_probs': probs})

def print_top_words(n_top_words, model, feature_names):
    
    for topic_idx, topic in enumerate(model.components_):
        norm_topic = np.divide(topic, sum(topic))
        
        print("Topic #%d:" % topic_idx)
        print(" ".join([(feature_names[i] + '(%0.3f' %norm_topic[i] + ')')
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
    print()

# get top words learnt by a model
def top_words_df(n_top_words, model, feature_names):
    res = pd.DataFrame({'topic':[], 'top_words':[], 'word_probs':[]})
    for t_idx, topic in enumerate(model.components_):
        top_words_of_topic = get_top_words(n_top_words, word_dist=topic, feature_names=feature_names)
        topic_idx = np.ones(n_top_words)*(t_idx+1)
        tmp = pd.concat([pd.DataFrame({'topic_idx': topic_idx }), top_words_of_topic], axis=1)
        res = pd.concat([res, tmp])
        
    return res[:][['topic_idx', 'top_words', 'word_probs']] # re-order columns as desired

## Similarity scores ====================================
def skillSim(p1, p2):
    skills1, skills2 = set(p1['occur_skills'].split(',')), set(p2['occur_skills'].split(','))
    intersection, union = skills1.intersection(skills2), skills1.union(skills2)
    return len(intersection)/float(len(union))

# Main workhorse, return sims in either data frame or matrix format, 
# output format is controlled by out_fmt
def pairwiseSim(posts, doc_topic_distr, out_fmt='data_frame', verbose=True):
    
    def topicSim(i, j, df):
        idx1, idx2 = getIndex(i, df), getIndex(j, df)
        d1, d2 = doc_topic_distr[idx1, :], doc_topic_distr[idx2, :]
        topic_sim = 1 - np.sqrt(JSD(d1, d2))
        return topic_sim

    def skillSimOfRows(i, j, df):
        """
        Jaccard similarity between 2 posts at rows i and j of given df
        """
        
        p1, p2 = df.iloc[i], df.iloc[j]
        return skillSim(p1, p2)
        # employer = p1['employer_name']
        # job_id1, job_id2 = p1['job_id'], p2['job_id']

    def simPair(i, j, df):
        topic_sim = topicSim(i, j, df)
        skill_sim = skillSimOfRows(i, j, df)

        # meta-info
        idoc, jdoc = df.iloc[i], df.iloc[j]
        job_id1, job_id2 = idoc['job_id'], jdoc['job_id']
        employer1, employer2 = idoc['employer_name'],jdoc['employer_name']
        skills1, skills2 = idoc['occur_skills'], jdoc['occur_skills']
        # doc1, doc2 = idoc['doc'], jdoc['doc']

        res = pd.DataFrame({'job_id1': job_id1, 'job_id2': job_id2, 
                            'topic_sim': round(topic_sim, 3), 'skill_sim': round(skill_sim, 2),
                            'skills1': skills1, 'skills2': skills2,
                            'employer1': employer1, 'employer2': employer2}, index=[1]) # 'doc1': doc1, 'doc2': doc2
        return res

    def sim2Subseq(i, df, out_fmt='lists'):
        """Similarity scores bw a post at row i with subseq posts in df"""

        n_doc = df.shape[0]
        # If there are actually subseq posts
        if (i <= n_doc-2): 
            if (i % 50 == 0) and verbose:
                print('\t {} posts and counting...'.format(i))
            
            if (out_fmt == 'data_frame'):
                frames = [simPair(i, j, df) for j in range(i, n_doc)] # i+1
                res = pd.concat(frames).reset_index(); del res['index']
                return res

            if (out_fmt == 'lists'):
                topic_sims = [topicSim(i, j, df) for j in range(i, n_doc)] # i+1
                skill_sims = [skillSimOfRows(i, j, df) for j in range(i+1, n_doc)]
                return pd.DataFrame({'topic_sim': topic_sims, 'skill_sim': skill_sims})
        pass

    def simMat(posts, level='topic'):
        n_post = posts.shape[0]
        sims = np.zeros(shape=(n_post, n_post))
        for i in xrange(n_post):
            sims[i, i] = 1
            # j < i
            for j in xrange(i): 
                sims[i, j] = sims[j, i]
            # j > i (only exist if i is not the last post, ie n_post-1)
            if (i < n_post-1) :
                if (level == 'topic'):
                    sims[i, (i+1):n_post] = sim2Subseq(i, posts, out_fmt='lists')['topic_sim']
                if (level == 'skill'):
                    sims[i, (i+1):n_post] = sim2Subseq(i, posts, out_fmt='lists')['skill_sim']

        return sims

    def simDF(posts):
        n_post = posts.shape[0]
        # t0 = time()
        frames = [sim2Subseq(i, posts, out_fmt='data_frame') for i in range(n_post)]
        # print('Done after %.1fs' %(time() - t0))
        return pd.concat(frames).reset_index()

    # n_post = posts.shape[0]
    # print('Computing pairwise similarity scores among {} job posts...'.format(n_post))
    # print('each post is compared with subseq posts...')
    
    if (out_fmt == 'data_frame'):
        return simDF(posts)

    if (out_fmt == 'matrix_topic_sim'):
        return simMat(posts, level='topic')

    if (out_fmt == 'matrix_skill_sim') :
        return simMat(posts, level='skill')

def rmBadPosts(posts, title):
    thres = posts.n_skill.quantile(.25)
    # print('Removed 1st quarter of posts of {}, each having <= {} skills'.format(title, int(thres)))
    return posts[posts.n_skill > thres]                

def sampleIfMoreThan(max_n_post, t, post_df):
    posts = post_df[post_df.title == t]
    return posts if len(posts) <= max_n_post else posts.sample(max_n_post)

def cachePosts(titles, post_df):
    max_n_post = 100
    res = {t: sampleIfMoreThan(max_n_post, t, post_df) for t in titles}
    print('Done caching sampled posts for titles with more than {}'.format(max_n_post))
    return res

def crossSimScores(posts1, posts2, doc_topic_distr, verbose=False):
    """
    Return cross sims (topic_sim and skill_sim) bw posts in 2 sets posts1 and posts2
    """
    def sims(p1, p2):
        idx1, idx2 = p1.name, p2.name
        d1, d2 = doc_topic_distr[idx1, :], doc_topic_distr[idx2, :]
        topic_sim = 1 - np.sqrt(JSD(d1, d2))
        skill_sim = skillSim(p1, p2)

        res = pd.DataFrame({'job_id1': p1.job_id, 'job_title1': p1.title, 'employer1': p1.employer_name,
                            'job_id2': p2.job_id, 'job_title2': p2.title, 'employer2': p2.employer_name,
                            'topic_sim': topic_sim, 'skill_sim': skill_sim,
                            'skills1': p1.occur_skills, 'skills2': p2.occur_skills},
                            index=[1])

        return res
    
    # global count; count = 0
    def sims2Set(p, posts):
        n_post = posts.shape[0]
        frames = [sims(p, posts.iloc[i]) for i in xrange(n_post)]
        # global count;  count += 1
        # if (count % 10 == 0) and verbose:
        #     print('%d posts and counting...' %count)

        return pd.concat(frames)

    n1 = posts1.shape[0]; n2 = posts2.shape[0]
    
    frames = [sims2Set(posts1.iloc[i], posts2) for i in xrange(n1)]
    res = pd.concat(frames);
    return res

def postSimScore(posts1, posts2, doc_topic_distr):
    ##     Rm lousy posts with too few skills from both sets
    #     posts1 = rmBadPosts(posts1, t1)
    #     posts2 = rmBadPosts(posts2, t2)
    
    n1, n2 = posts1.shape[0], posts2.shape[0]
    if (n1 > 0) and (n2 > 0):
        res = crossSimScores(posts1, posts2, doc_topic_distr, verbose=False)
        topic_sim = round(res['topic_sim'].mean(), 3)
        return topic_sim  # return res
    return np.nan

def titleSim(t1, t2, doc_topic_distr, df=None, cached_posts=None, verbose=False):
        
    # posts1 = df[df.title == t1] 
    # posts2 = df[df.title == t2]
    
    posts1 = cached_posts[t1]
    posts2 = cached_posts[t2]

    if verbose: 
        n1, n2 = posts1.shape[0], posts2.shape[0]
        print('\t{} ({} posts) vs. {} ({} posts)'.format(t1, n1, t2, n2))
    return postSimScore(posts1, posts2, doc_topic_distr)

def sims2SubseqTitle(i, titles, doc_topic_distr, cached_posts=None, verbose=False):
    '''
    @param  i: index of the focus title
    @param  titles
    @return topic sims of i-th title with its sub seq titles in the given titles
    '''
    n_title = len(titles); focus_title = titles[i]
    sub_seq_titles = titles[i+1 : n_title]
    res = pd.DataFrame({'t1': sub_seq_titles, 't2': focus_title})
    res['topic_sim'] = res['t1'].apply(titleSim, t2=focus_title, 
                                       doc_topic_distr=doc_topic_distr,
                                       cached_posts=cached_posts, verbose=verbose)
    
    print('\t Calculated sims of {} to all subseq titles'.format(focus_title))
    return res

def calSims4Batch(b, size, titles, doc_topic_distr, cached_posts):
    start = b*size; end = start + size
    t0 = time()
    frames = [sims2SubseqTitle(i, titles, doc_topic_distr, cached_posts) for i in range(start, end)]
    elapse = round(time() - t0, 1)
    print('\tFinished sim cals for a batch of {} job titles in {}s'.format(size, elapse))
    return frames

def saveBatch(b, res, tmp_dir):
    res = res.reset_index(); del res['index']
    res.to_csv(tmp_dir + 'b{}.csv'.format(b), index=False)
    print('\t Saved results of batch {}'.format(b))

def calAndSaveSims4Batch(b, bsize, titles, doc_topic_distr, cached_posts, tmp_dir):
    frames = calSims4Batch(b, bsize, titles, doc_topic_distr, cached_posts)
    res = pd.concat(frames)
    saveBatch(b, res, tmp_dir)
    return res

def simsAmong(titles, doc_topic_distr, df, verbose=False, bsize=50, tmp_dir=''):
    n_title = len(titles)
    msg = '# job titles: {}. For job titles with > 100 posts, only sample 100 posts.'
    print(msg.format(n_title))
    
    if n_title > 1:
        cached_posts = cachePosts(titles, df) # s.t. we do not have to repeat sampling
        
        # if no. of title is large, it is better to
        # cal for each batch and save defensively to avoid loss of results
        n_batch = n_title/bsize; remains = n_title % bsize
        
        frames = [calAndSaveSims4Batch(b, bsize, titles, doc_topic_distr, cached_posts, tmp_dir) 
                    for b in xrange(0, n_batch)]
        
        return pd.concat(frames)
    
    else: # return an empty data frame instead of None
        return pd.DataFrame({'t1': [], 't2': [], 'topic_sim': []})

def buildTopkFrom(rel_sims, k, job_title):
    rel_sims.sort_values('topic_sim', ascending=False, inplace=True)
    tmp = rel_sims.head(k).copy()
    # if focus job title is the 2nd column, swap it to 1st column
    part1 = tmp[tmp.t1 == job_title]
    part2 = tmp[tmp.t2 == job_title]
    topk = part1.append(swapCols('t1', 't2', part2))
    
    topk['title_n_sim'] = pasteCols('t2', 'topic_sim', topk)
    return topk
    # return ', '.join(topk['title_n_sim'])

def topkByFunction(job_title, k, func_sims):
    q = 't1 == "{}" or t2 == "{}"'.format(job_title, job_title)
    rel_sims = func_sims.query(q)
    return buildTopkFrom(rel_sims, k, job_title)

## Funcs for filtering ====================================
def titlesIn(domain, title_df):
    return title_df.title[title_df.domain == domain].unique()

def titlesWithFunc(pri_func, title_df):
    sub_df = title_df[title_df.pri_func == pri_func]
    sub_df = sub_df.sort_values('n_post', ascending=False)
    return sub_df.title.unique()

def titlesHavingAtLeast(records, min_post):
    return list(records[records.n_post >= min_post]['title'])

## Visualization ====================================
def getPostsInPairs(pairs, df):
    job_ids = set(pairs.job_id1.append(pairs.job_id2))
    posts = df[df.job_id.isin(job_ids)]
    print('# posts retrieved: %d' %posts.shape[0])
    return posts

def pretty(employer_name='Millennium Capital Management (Singapore) Pte. Ltd.'):
    name = employer_name.title()
    name = name.replace('Pte.','').replace('Ltd.','').replace('(Singapore)','').replace('Job-', '')
    return name.strip()

def plotDist(post, doc_topic_distr, labels):
    """
    @param      post             
    @param      doc_topic_distr  contains cluster distributions of all posts
    @return     Bar chart of the cluster distribution of given post (bars at x locs)
    """
    n_topic = doc_topic_distr.shape[1] 
    topic_idx = np.arange(1, n_topic + 1)
    
    if len(topic_idx) != len(labels):
        print('# xticks ({}) != # labels ({})!'.format(len(topic_idx), len(labels)))
        pass

    if len(topic_idx) == len(labels):
        job_id, employer, job_title = pretty(post['job_id']), pretty(post['employer_name']), post['title']
        idx = post.name #  post['index']
        probs = doc_topic_distr[idx, :]

        bars = plt.bar(topic_idx, probs)  # width = 0.3
        plt.xticks(topic_idx, labels, rotation=45)
        plt.xlim(0, n_topic + 1)
        plt.grid(True)

        plt.title(job_id + '(' + job_title + ' at ' + employer + ')')
        # print('x = {}'.format(x))
        return bars

def topicDists(p1, p2, doc_topic_distr, labels):
    fig, axes = plt.subplots(2, sharex=True, figsize=(6,6))
    
    plt.subplot(211)
    bars1 = plotDist(p1, doc_topic_distr, labels)

    plt.subplots_adjust(hspace=.3)  
    
    plt.subplot(212)
    bars2 = plotDist(p2, doc_topic_distr, labels)
    
    plt.xlabel('Skill Clusters', fontsize=16)

    # hide x-ticks of 1st subplot (NOTE: this kind of fine tune need to be done last, why?)
    plt.setp(fig.axes[0].get_xticklabels(), visible=False)
    return fig

def vizDists4Pair(row, df, doc_topic_distr, labels):
    """
    @brief      Plot cluster distributions of the pair of posts stored at given row (in a df of post sims)
    @param      row   
    @param      df     {storing posts and their indices in the matrix doc_topic_distr}
    @return     {2 bar charts of the cluster distributions, sharing x axis}
    """
    p1 = df[df.job_id == row.job_id1].iloc[0]; p2 = df[df.job_id == row.job_id2].iloc[0]

    fig = topicDists(p1, p2, doc_topic_distr, labels)
    topic_sim, skill_sim = round(row.topic_sim, 3), round(row.skill_sim, 3)
    title = 'Topic similarity: {}, skill similarity {}'.format(topic_sim, skill_sim)
    fig.suptitle(title, fontsize=16)
    return fig

topic_df = pd.read_csv('d:/larc_projects/job_analytics/results/lda/20_topics.csv')
labels = map(str.upper, topic_df['label'])
def vizPostPair(i, sim_df, post_df, doc_topic_distr, labels, abbv_title=''):
    row = sim_df.iloc[i]
    fig = vizDists4Pair(row, post_df, doc_topic_distr, labels)
    fig.savefig(RES_DIR + 'fig/{}_p{}.pdf'.format(abbv_title, i+1))
    plt.show(); plt.close()

# May not be needed anymore
def vizTopicDists(posts, doc_topic_distr, figsize):
    """
    Plot cluster distributions of posts
    """

    n_post = posts.shape[0]
    fig, axarr = plt.subplots(n_post, sharex='col', sharey='row', figsize=figsize) # sharex=True

    n_group = 2; group_size = n_post/n_group
    lasts = range((group_size-1)*n_group, n_post)

    for i in range(n_post):
        plt.subplot(group_size, n_group, p+1)
        plotDist(posts.iloc[i], doc_topic_distr)
        # Show xtick labels and x-label only for the last subplot in each group
        if p in lasts:
            plt.xticks(x, labels, rotation=45)
            plt.xlabel('Skill Clusters', fontsize=16)

        # Show ylabel at the middle
        # if p==(n_post/2 - 1): 
        #     plt.ylabel('Probability', fontsize=16)

    ## Fine tune the fig    
    fig.subplots_adjust(hspace=.5)
    # In each group, hide xticks on all subplots except the last one 
    hide_xticks(fig, lasts)
    
    return fig

def vizSkillSim(sim_df, ax, sci_fmt=True, fontsize=16):
    skill_sim = sim_df.skill_sim
    
    hist = ax.hist(skill_sim)
    ax.grid(True)
    
    skill_sim_mean, skill_sim_std = round(skill_sim.mean(), 2), round(skill_sim.std(), 2)
    xl = 'Skill Similarity\n' + r'$(\mu: {}, \sigma: {})$'.format(skill_sim_mean, skill_sim_std)
    plt.xlabel(xl, fontsize=fontsize) 
    plt.ylabel('Count', fontsize=fontsize)
    if sci_fmt:
        ax.yaxis.set_major_formatter( mtick.FormatStrFormatter('%.1e') )
    # setAxProps(ax, fontproperties)
    return hist

def vizTopicSim(sim_df, ax=None, sci_fmt=True, fontsize=16):
    topic_sim = sim_df.topic_sim
    
    if ax:
        ax.hist(topic_sim)
        ax.grid(True)
    else:
        plt.hist(topic_sim)
        plt.grid(True)
    
    topic_sim_mean, topic_sim_std = round(topic_sim.mean(), 3), round(topic_sim.std(), 3)
    xl = 'Topic Similarity\n' + r'$(\mu: {}, \sigma: {})$'.format(topic_sim_mean, topic_sim_std)
    plt.xlabel(xl, fontsize=fontsize)
    plt.ylabel('# pairs of job titles', fontsize=fontsize)
    if sci_fmt:
        ax.yaxis.set_major_formatter( mtick.FormatStrFormatter('%.1e') )
        # setAxProps(ax, fontproperties)
    
def plotSimDists(sim_df, figsize=(10, 5), sci_fmt=True):
    """
    @param      sim_df     
    @return     2 hists of topic_sim and skill_sim (in sim_df) of job posts
    """

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=figsize)
    fontsize = 16; fontweight = 'bold'
    fontproperties = {'weight' : fontweight} # 'family':'sans-serif','sans-serif':['Helvetica'], 'size' : fontsize

    ax = plt.subplot(1,2,1)
    skill_sim_hist = vizSkillSim(sim_df, ax, sci_fmt)

    plt.subplots_adjust(wspace=.5, bottom=.15) # top=.9
    
    ax = plt.subplot(1,2,2)
    topic_sim_hist = vizTopicSim(sim_df, ax, sci_fmt)
    
    return fig

def viz(sims):
    fig, ax = plt.subplots()
    vizTopicSim(sims, ax)
    fig.subplots_adjust(bottom=.2)
    return fig

def vizJobPostDist(by_n_post):
    """
    @param      by_n_post: group job titles by their number of posts
    @return     The 2 distributions of job posts in job titles 
                before and after title standardization
    """

    fig = plt.figure()
    n_post_vals = by_n_post.n_post
    plt.scatter(x=n_post_vals, y=by_n_post.n_title, marker='o', c='b')
    plt.scatter(x=n_post_vals, y=by_n_post.n_title_after_std, marker='x', c='r')
    plt.loglog()
    plt.xlabel('# job posts'); plt.ylabel('# job titles')
    plt.xlim(min(n_post_vals), max(n_post_vals)*10)
    plt.grid(True)
    plt.legend(['Before title standardization', 'After title standardization'])
    return fig

## Eval against SkillFuture framework
def skillFreq(posts):
    skills_in_posts = ','.join(posts.occur_skills); ls_skills = skills_in_posts.split(',')
    c = Counter(ls_skills)

    skill_df = pd.DataFrame({'skill': c.keys(), 'freq': c.values()})
    return skill_df.sort_values('freq', ascending=False)

def getTitleStats(posts, titles=None):
    """
    @param      titles, None if we just want to get stats for all titles in
                the posts instead of a specific set of titles
    @param      posts
    @return     The statistics for the titles in the given posts, not in whole ds
    """
    by_title = posts.groupby('title')

    tmp = by_title.agg({'n_skill': np.mean, 'job_id': len, 'employer_id': 'nunique'})
    tmp = tmp.reset_index();
    tmp = tmp.rename(columns={'job_id': 'n_post', 'n_skill': 'avg_n_skill', 
                    'employer_id': 'n_employer'})

    if not titles:
        return tmp.sort_values('n_post', ascending=False).round(1)
    if titles:
        tmp = tmp[tmp.title.isin(titles)]
        return tmp.sort_values('n_post', ascending=False).round(1)

# def freq(sk, skill_sets):
#     count = 0;
#     for ss in skill_sets:
#         if sk in ss: count += 1
    
#     return count

## Others
def findOutliers(res):
    lif, lof = getLowerFences(res['topic_sim'])
    sus_outliers = res.query('topic_sim < {}'.format(lif)).query('topic_sim > {}'.format(lof))
    outliers = res.query('topic_sim < {}'.format(lof))
    return [sus_outliers, outliers]

def analyzeOutliers(res):
    outliers = pd.concat(findOutliers(res))
    return outliers.sort_values('topic_sim')

def trainNMF(tfidf_mat):
    pass

def plotMetrics(train_metric, test_metric, model_name):    
    fig = plt.figure(figsize=(6,6))
    plt.subplot(2, 1, 1)
    plt.plot(ks, train_metric)
    plt.xlabel('No. of topics')
    plt.ylabel(r'$\| X_{train} - W_{train} H \|_2$')
    plt.title('Error of {} on train set'.format(model_name))
    plt.grid(True)
    plt.xticks(ks)
    plt.subplots_adjust(wspace=.5, hspace=.5)

    plt.subplot(2, 1, 2)
    plt.plot(ks, test_metric)
    plt.xlabel('No. of topics')
    plt.ylabel(r'$\| X_{test} - W_{test} H \|_2$')
    plt.title('Error of {} on test set'.format(model_name))
    plt.grid(True)
    plt.xticks(ks)

    plt.show()
    return fig    