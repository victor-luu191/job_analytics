# Job and Skill Analytics

This repo contains source codes for analyzing data of job postings to gain insights on jobs and skills required for different jobs and sectors. The codes are written in `Jupyter` notebooks and all helper functions are centralized in `ja_helpers.py` module. In addition, there are some more helpers in `my_util.py`, `stat_helpers.py` and `parse_funs.py`.

## Main components
Main components handle the following steps in the pipeline: i) __Preprocess job posts__ and meta-data, ii) __Extract features__ for topic and matrix factorization (MF) models, iii) __Cluster skills__ using the models, iv) __Connect job titles__ based on their topic similarity and v) __Determine consistency__ among job posts of a given job title.

Following are details on each component.

1. __Preprocess__. 
Maily handled by `preprocess` and `parse_title` notebooks. In addition, there is also a notebook `filter_dups` for filtering __duplicated job posts__, which are basically the same posting but re-posted several times. This happens in two cases: i) different branches of a group/big company post the same posting or ii) a company and its representing recruiting agencies post the same posting.

`preprocess` handles the following tasks:
  + filter posts containing only 1 skill
  + filter skills occuring in only 1 post
  + filter stop-word skills, e.g. business

+ In: job post texts (with html tags and punctuations already removed by beautifulsoup) and skill vocabulary.
+ Out: filtered posts (saved in clean/job_posts.csv) and filtered skill vocabulary.

In addition, there is also one part for cleaning data on employers. Input: raw/employers.csv file containing employer data such as UENs and employer names. Output: clean employer data stored in clean/employers.csv file.

`parse_title` parses each job titles into separate components: position, domain and primary function (some job titles may also have secondary function). This parsing serves two goals: 
  + grouping job titles by their domains or functions (or by position).
  + standardizing job titles to unify different forms of the same job title e.g. Software Engineer and Engineer, Software will be unified as Software Engineer.
The script uses the parser from https://jobsense.sg/api/get/job-title-parser/.

+ In: __unique titles__ of job posts in either data/clean/doc_index_filter.csv or any table of job posts with titles included.
+ Out: a table (in parsed_titles.csv) where each row contains a job title together with its __components__ obtained from parsing. Some titles which could not be parsed are saved in invalid_titles.csv file.

2. __Extract features__ required by topic and MF models.

Each job post is regarded as a document and each skill is an entry in the vocabulary of skills.
The features for LDA are represented by a document-skill matrix whose each entry _e(d, s)_ is occurrence count of skill _s_ in document _d_. As the skills can be uni-, bi- or tri-grams, there is one important difference from counting uni-gram words in documents: __counts of certain skills can be over-estimated__!!! For example, count of _programming_, as a skill itself, can be inflated as it also appears in other skills including 'Java programming', 'Python programming' and so on.

The script in `extract_feat.ipynb` handles this problem by first counting occurence of longer n-grams. Once the couting is done, it removes longer n-grams from documents and go on to count shorter n-grams. The removal gets rid of the over-estimation.

+ In: filtered job posts and skill vocabulary.
+ Out: document-skill matrix stored in clean/doc_skill.mtx file. In this matrix, index of documents (i.e., which job posts go with which row) are saved in clean/doc_index.csv or clean/doc_index_filter.csv; similarly index of skills are stored in clean/skill_index.csv. These indices should be kept unchanged for correct lookups later.

3. __Cluster skills__ using LDA.

I adopt the module for LDA in scikit-learn. The script is in `cluster_skill.ipynb`.
+ In: no. of topics _k_ and document-skill matrix in clean/doc_skill.mtx file.
+ Out: all outputs in this step are stored in results/lda in G drive. For each _k_, return a matrix representing topics as word distributions (saved in {_k_}topic_word_dist) and a matrix representing topic distributions of documents (saved in doc_{k}_topic_distr.mtx). Due to storage limitations, the matrices are compressed in folders word_dists.zip and doc_topic_dists.zip (Only doc_20topic_distr.mtx are unzipped for convenience).

__Note__: index of documents in doc_{k}_topic_distr.mtx and doc_skill.mtx should match with each other and with doc_index.csv.

4. __Connect job titles__.

Given topic distribution of each job post learnt by LDA, we can compute topic similarity score between any two job posts as the similarity of their topic distributions. 

Given two job titles t and t' with their sets of job posts P(t) and P(t'), we define similarity between t and t' as the average of pairwise similarity scores of pairs of posts (p, p') where p and p' belong to P(t) and P(t') respectively.

Similarity scores between pairs of job titles under the same domain are precomputed by the script in `sim_by_domain.ipynb`.
+ In: parsed titles grouped by domains, doc_index_filter.csv (with document indices for lookup and standard job titles) and doc_20topic_distr.mtx (to retrieve topic distributions of documents). 
+ Out: for each domain, return pairwise similarities of job titles in that domain. All the similarity results are saved in bydomain_sims.rar.

Similarly, similarity scores between pairs of job titles with same primary function are precomputed in `sim_by_func.ipynb`.
+ In: same as above but now titles are grouped by primary functions.
+ Out: for each function, return pairwise similarities of job titles in that function. All the similarity results are saved in byfunction_sims.rar.

5. __Determine consistency among job posts.__

We define consistency score for a given set of job posts as the average of pairwise similarity scores over all pairs of posts from the set.

Given a job title, we then define its consistency score as the consistency score of the set of its posts. The consistency score allows us to profile a job title as _general_ or _niche_ job title (e.g. Manager is general while Software Engineer is niche). Computing consistency scores is done in `job_consistency.ipynb`.
