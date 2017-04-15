# Job and Skill Analytics

This repo contains source codes for analyzing data of job postings to gain insights on jobs and skills required for different jobs and sectors. The codes are written in `Jupyter` notebooks and all main functions are centralized in `ja_helpers.py` module.

## Main components
Main components handle the following steps in the pipeline: i) __Preprocess job posts__ and meta-data, ii) __Extract features__ for topic and matrix factorization (MF) models, iii) __Cluster skills__ using the models, iv) __Connect job titles__ based on their topic similarity and v) __Determine consistency__ among job posts of a given job title.

Following are details on each component.

1. Preprocess. 
This step is handled by `preprocess` and `parse_title` notebooks.

`parse_title` parses each job titles into separate components: position, domain and primary function (some job titles may also have secondary function). This parsing serves two goals: 
  + grouping job titles by their domains or functions (or by position).
  + standardizing job titles to unify different forms of the same job title e.g. Software Engineer and Engineer, Software will be unified as Software Engineer.
The script uses the parser from https://jobsense.sg/api/get/job-title-parser/.

`preprocess` handles the following tasks:
  + filter posts containing only 1 skill
  + filter skills occuring in only 1 post
  + filter stop-word skills, e.g. business
In addition, there is also one part for cleaning data on employers.

2. Extract features required by topic and MF models.
Each job post is regarded as a document and each skill is an entry in the vocabulary of skills.
The features for LDA are represented by a document-skill matrix whose each entry _e(d, s)_ is occurrence count of skill _s_ in document _d_. As the skills can be uni-, bi- or tri-grams, there is one important difference from counting uni-gram words in documents: __counts of certain skills can be over-estimated__!!! For example, count of _programming_, as a skill itself, can be inflated as it also appears in other skills including 'Java programming', 'Python programming' and so on.

The script in `extract_feat.ipynb` handles this problem by first counting occurence of longer n-grams. Once the couting is done, it removes longer n-grams from documents and go on to count shorter n-grams. The removal gets rid of over-estimation.

3. Cluster skills using LDA.
We adopt the LDA module in scikit-learn.

4. Connect job titles.
Given topic distribution of each job post learnt by LDA, we can compute topic similarity between any two job posts as the similarity of two topic distributions.

5. Determine consistency among job posts.
