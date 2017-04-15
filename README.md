# Job and Skill Analytics

This repo contains source codes for analyzing data of job postings to gain insights on jobs and skills required for different jobs and sectors.

## Main components
Main components handle the following steps in the pipeline: i) Preprocess job posts and meta-data, ii) Extract features for topic and matrix factorization (MF) models, iii) Run the models, iv) Connect job titles based on their topic similarity and v) Determine consistency among job posts by their topic similarity.

Following are details on each component.
1. Preprocess: 
This step include the following tasks:
  + filter posts containing only 1 skill
  + filter skills occuring in only 1 post
  + filter stop-word skills, e.g. business
In addition, there is also one part for cleaning data on employers.

2. Extract features required by topic and MF models:

Each job post is regarded as a document and each skill is an entry in vocabulary. 
The features are represented by a document-skill matrix whose each entry $e(d, s) $ is either occurrence count of skill s in document d.
