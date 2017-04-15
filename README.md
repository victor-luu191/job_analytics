# Job and Skill Analytics

This repo contains source codes for analyzing data of job postings to gain insights on jobs and skills required for different jobs and sectors. The codes are written in `Jupyter` notebooks and the main functions are centralized in `ja_helpers.py` module.

## Main components
Main components handle the following steps in the pipeline: i) __Preprocess job posts__ and meta-data, ii) __Extract features__ for topic and matrix factorization (MF) models, iii) __Cluster skills__ using the models, iv) __Connect job titles__ based on their topic similarity and v) __Determine consistency__ among job posts of a given job title.

Following are details on each component.

1. Preprocess. 
This step is handled by `preprocess` and `parse_title` notebooks.

`parse_title` parses each job titles into separate components: position, domain and primary function (some job titles may also have secondary function). The script uses the parser from https://jobsense.sg/api/get/job-title-parser/.

`preprocess` handles the following tasks:
  + filter posts containing only 1 skill
  + filter skills occuring in only 1 post
  + filter stop-word skills, e.g. business
In addition, there is also one part for cleaning data on employers.

2. Extract features required by topic and MF models.
Each job post is regarded as a document and each skill is an entry in vocabulary. 
The features for LDA are represented by a document-skill matrix whose each entry _e(d, s)_ is occurrence count of skill _s_ in document _d_.

3. Cluster skills using LDA.
We adopt the LDA module in scikit-learn.

4. Connect job titles.
Given topic distribution of each job post learnt by LDA, we can compute topic similarity between any two job posts as the similarity of two topic distributions.

5. Determine consistency among job posts.
