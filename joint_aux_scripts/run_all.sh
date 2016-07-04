dataset=$1 # E.g. /mnt/data/corpora/ted-cldc.
word_vectors=$2 # E.g. /mnt/data/corpora/word_vectors/glove/glove.840B.300d.txt
num_epochs_embeddings=50
regularization_type=l1 #l2 #l1
mu=1e-9 # 1e-9 for L1, 0.001 for L2.
num_epochs_task=100
lambda_task=0.1 # 0.1 for L1, 0.01 for L2.
lambda_task_bow=0.00001 # or 0.000001
suffix_embeddings=reg-${regularization_type}_mu-${mu}_epochs-${num_epochs_embeddings}
source_embeddings_file=P_${suffix_embeddings}.txt
target_embeddings_file=Q_${suffix_embeddings}.txt

echo '--------------------------------------------------------'
echo 'Creating parallel dataset for training the embeddings...'
echo '--------------------------------------------------------'
#python create_parallel_dataset.py ${dataset}

echo '--------------------------------------------------------'
echo 'Selecting the English word vectors present in the data...'
echo '--------------------------------------------------------'
#python select_word_vectors.py ${word_vectors} ted_vocab.source \
#    > word_vectors.txt

echo '--------------------------------------------------------'
echo 'Solving the joint optimization to obtain multilingual embeddings...'
echo '--------------------------------------------------------'
#python compute_joint_embeddings.py ted.source ted.target ted_vocab.source \
#    ted_vocab.target word_vectors.txt ${regularization_type} ${mu} \
#    ${num_epochs_embeddings} ${source_embeddings_file} \
#    ${target_embeddings_file}


echo '--------------------------------------------------------'
echo 'Interactively checking the quality of the embeddings...'
echo '--------------------------------------------------------'
#python compute_closest_embeddings.py \
#    ${source_embeddings_file} ${target_embeddings_file}


echo '--------------------------------------------------------'
echo 'Creating multilingual train/evaluation datasets...'
echo '--------------------------------------------------------'
python create_multilabel_datasets.py ${dataset} \
    ${source_embeddings_file} ${target_embeddings_file}

#for language in de es
for language in ar de es fr it nl pb pl ro ru tr zh
do
    echo '--------------------------------------------------------'
    echo "Training/evaluating cross-lingual system for ${language}..."
    echo '--------------------------------------------------------'
    python train_multilabel_classifier_lbfgs.py logistic ${num_epochs_task} \
        ${lambda_task} ted.en-${language}.train ted.${language}-en.test 1 1 >& \
        log_en-${language}_${suffix_embeddings}_lambda-${lambda_task}_epochs-${num_epochs_task}.txt

    echo '--------------------------------------------------------'
    echo "Training/evaluating English system via ${language}..."
    echo '--------------------------------------------------------'
    python train_multilabel_classifier_lbfgs.py logistic ${num_epochs_task} \
        ${lambda_task} ted.en-${language}.train ted.en-${language}.test 1 1 >& \
        log_mono_en_${language}_${suffix_embeddings}_lambda-${lambda_task}_epochs-${num_epochs_task}.txt

    echo '--------------------------------------------------------'
    echo "Training/evaluating monolingual system for ${language}..."
    echo '--------------------------------------------------------'
    python train_multilabel_classifier_lbfgs.py logistic ${num_epochs_task} \
        ${lambda_task} ted.${language}-en.train ted.${language}-en.test 1 1 >& \
        log_mono_${language}_${suffix_embeddings}_lambda-${lambda_task}_epochs-${num_epochs_task}.txt

    echo '--------------------------------------------------------'
    echo "Training/evaluating BOW English system via ${language}..."
    echo '--------------------------------------------------------'
    #python train_multilabel_classifier_lbfgs.py logistic ${num_epochs_task} \
    #    ${lambda_task_bow} ted.bow.en-${language}.train ted.bow.en-${language}.test 0 1 >& \
    #    log_bow_mono_en_${language}_${suffix_embeddings}_lambda-${lambda_task_bow}_epochs-${num_epochs_task}.txt

    echo '--------------------------------------------------------'
    echo "Training/evaluating BOW monolingual system for ${language}..."
    echo '--------------------------------------------------------'
    #python train_multilabel_classifier_lbfgs.py logistic ${num_epochs_task} \
    #    ${lambda_task_bow} ted.bow.${language}-en.train ted.bow.${language}-en.test 0 1 >& \
    #    log_bow_mono_${language}_${suffix_embeddings}_lambda-${lambda_task_bow}_epochs-${num_epochs_task}.txt

done
