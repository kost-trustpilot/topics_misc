# Stand-alone script to compute:
# 1) N x L similarity scores between L topics and N sentences
# 2) L x L correlation matrix for L topic series of similarity scores
# 3) L x 7 statistics matrix for L topics, incl nearest neighbors lists

# Required modules:
# pandas, numpy, scikit-learn, sentence_transformers (only if compute_topic_embeddings==True)

import numpy as np
import pandas as pd
import time
from sklearn.metrics.pairwise import cosine_similarity

# select pipeline steps
compute_topic_embeddings = False
compute_topic_similarities = True
compute_correlations = True
compute_stats = True

# data scope (obs: 9k labels will generate ~15GB of similarity data for each 1mln of sentences, if integer16 encoded)
max_sentences = 3200000 # None -> include all (~16mln)
max_topics = None # None -> include them all (~9k)
n_batches = 160

# if restarting job, specify batch >0
batch_start = 0

min_text_length = 4
encoder_device = 'cpu'
encoder_batch_size = 100
encoder_float_type = 'float32'
bert_model = 'stsb-xlm-r-multilingual'
#similarity_max_cpu = 12
#similarity_batch_size = 500


# GCS input directory and local output directory
gcs_dir = r'gcs://datascience-models-staging/pi_topics/candidate_selection'
data_dir = r'/home/kost/python_data/trustpilot/topic/candidates'

# INPUTS
# should contain list of candidate topics:
topic_cand_filename = '%s/topic_candidates_7sept.csv' % gcs_dir
# should contain [text, embeddings] basline dataset dataframe:
benchmark_dataset = r'gcs://datascience-models-staging/sentiment_sise_stse_datasets/STSE_v6/master/STSE_2010_2020_master.parquet'
# dataset batches:
benchmark_batches = []
for i in range(n_batches):
    benchmark_batches.append('%s.%d' % (benchmark_dataset, i))

# OUTPUTS
# output dataframe containing [text, embeddings] of candidate topics
topic_cand_embeddings_filename = '%s/topic_candidates_7sept.parquet'%data_dir
# output dataframe (batches) containing similarity scores (rows: sentences, columns: candidates) (N x L)
topic_sims_filename = '%s/topic_sims_int16.parquet' % (data_dir)
# output similarities batches
topic_sims_filenames = []
for i in range(n_batches):
    topic_sims_filenames.append('%s.%d' % (topic_sims_filename, i))
# output dataframe containing correlation matrix (L x L)
topic_corr_filename = '%s/topic_cand_corr_matrix.parquet' % data_dir
# output dataframe containing statistics and nearest neighbors (N x 7)
topic_stats_filename = '%s/topic_cand_stats.parquet' % data_dir    


df_corr = None
corr = None


# Run pipeline
if __name__ == '__main__':


    if compute_topic_embeddings:
        # Compute xlm-r embeddings for candidate labels
        from sentence_transformers import SentenceTransformer
        topics = pd.read_csv(topic_cand_filename, sep=',')
        bert_model_dir = bert_model.split('/')[-1]
        print('\n--- Encoding %d candidate topics with model: %s (device: %s)---\n' % (topics.shape[0], bert_model, encoder_device))
        model = SentenceTransformer(bert_model)
        texts = list(topics.text)
        print('\nEncoding topics..')
        embeddings = model.encode(texts, batch_size=encoder_batch_size, device=encoder_device, show_progress_bar=True)
        print('Saving topic embeddings..')
        topics['embeddings'] = embeddings.tolist()
        if encoder_float_type == 'float32':
            topics['embeddings'] = topics['embeddings'].apply(lambda x: np.array(x).astype('float32'))
        topics.to_parquet(topic_cand_embeddings_filename, compression=None, index=False)


    if compute_topic_similarities:
        # Compute similarity score (i.e. cosine_similarity) for each pair (sentence x candidate)
        topics = pd.read_parquet(topic_cand_embeddings_filename)
        topics = topics[['text', 'embeddings']]
        if max_topics is not None:
            topics = topics.sample(n=max_topics)

        print('Computing similarity scores for %d batches..' % (len(benchmark_batches)))
        for i, f in enumerate(benchmark_batches):
            if i >= batch_start:
                t_period = time.time()
                try:
                    batch = pd.read_parquet(f)
                except:
                    break

                if max_sentences is not None:
                    batch_frac = int(max_sentences/n_batches)
                    if batch_frac>batch.shape[0]:
                        batch_frac = batch.shape[0]
                    batch = batch.sample(n=batch_frac)
                if min_text_length is not None:
                    batch['n_words'] = batch.text.str.split(' ').str.len()
                    batch = batch[batch['n_words']>=min_text_length]
                batch = batch[['text', 'embeddings']]
                if batch.shape[0] == 0:
                    result = pd.DataFrame(columns=list(topics.text)).reset_index().rename(columns={'index': '_text_'})
                    result.to_parquet(topic_sims_filenames[i], compression=None, index=False)
                    continue

                t_period = time.time()

                sentence_vecs = np.array([np.array(vv) for vv in batch.embeddings.values])
                topic_vecs = np.array([np.array(vv) for vv in topics.embeddings.values])
                sims = np.round( cosine_similarity(sentence_vecs, topic_vecs) * 1000 ).astype('int16')
                result = pd.DataFrame(index=list(batch.text), columns=list(topics.text), data=sims)
                result = result.reset_index().rename(columns={'index':'_text_'})
                # sims = np.random.rand(batch.shape[0], topics.shape[0]).astype('float32')

                t_period = time.time() - t_period

                print('Batch %d/%d: computed (%dx%d) similarities in %.1f sec..' % (i, len(benchmark_batches), sims.shape[0], sims.shape[1], t_period))
                result.to_parquet(topic_sims_filenames[i], compression=None, index=False)


    if compute_stats or compute_correlations:
        # Load similarity batches
        sims = []
        for i in range(n_batches):
            print('Reading batch %d/%d..' % (i, n_batches))
            s = pd.read_parquet('%s/topic_sims_int16.parquet.%d' % (data_dir, i))
            if s.shape[0]>0:
                sims.append(s)
        sims = pd.concat(sims, axis=0)
        sims = sims.set_index('_text_')


    if compute_correlations:
        # Calculate pearson correlation coefficients for each candidate-candidate pair (correlation matrix L x L)
        print('Computing %dx%d correlations for series of length %d..' % (sims.shape[1], sims.shape[1], sims.shape[0])) 
        t1 = time.time()
        corr = np.corrcoef(sims.values, rowvar=False)
        t1 = time.time()-t1
        print('Correlation computing time: %.2f'%t1)
        df_corr = pd.DataFrame(index=list(sims.columns), columns=list(sims.columns), data=corr)
        df_corr.to_parquet(topic_corr_filename, compression=None, index=True)


    if compute_stats:
        # Calculate statistics (mean, std, min, max, quantiles) for each candidate similarity scores (statistics matrix N x 7)
        print('Computing %d statistics for series of length %d..' % (sims.shape[1], sims.shape[0])) 
        if df_corr is None or corr is None:
            df_corr = pd.read_parquet(topic_corr_filename)
            corr = df_corr.values
        t1 = time.time()
        df_stats = sims.describe().transpose()
        df_stats = df_stats.loc[df_corr.columns,:]
        # Attach list of nearest neighbor candidates (by correlation) -> max 20, correlation>=0.8
        corr_min = 0.8
        neighbors = []
        for t in range(df_stats.shape[0]):
            cors_idx = np.argsort(corr[t,:])[::-1]
            cors_vals = [str(corr[t,x]) for x in cors_idx[:20]]
            cors_labels = [df_corr.columns[x] for x in cors_idx[:20]]
            neighbors.append( ', '.join(['%s'%(cors_labels[ix]) for ix in range(len(cors_labels)) if float(cors_vals[ix])>=corr_min]))
        # Optionally sort
        #df_stats = df_stats.sort_values('mean', ascending=False)        
        t1 = time.time()-t1
        print('Statistics computing time: %.2f'%t1)
        df_stats.to_parquet(topic_stats_filename, compression=None, index=True)
