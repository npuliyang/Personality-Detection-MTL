from os.path import join, exists, split
import os
import numpy as np

def load_word2vec(model_type, vocabulary_inv, num_features=300):
    """
    loads Word2Vec model
    Returns initial weights for embedding layer.

    inputs:
    model_type      # GoogleNews / glove
    vocabulary_inv  # dict {str:int}
    num_features    # Word vector dimensionality
    """

    # model_dir = './data/glove.6B/'
    model_dir = "./drive/My Drive/Colab Notebooks/"
    # model_dir = "/media/yang/Ryan file/Ryan_Files/Doctor_Research/Research_TextData/Sentiment_analysis/Data/glove.6B/"

    if model_type == 'glove':
        model_name = join(model_dir, 'glove.6B.%dd.txt' % (num_features))
        # print(model_name)
        assert(exists(model_name))
        print('Loading existing Word2Vec model (Glove.6B.%dd)' % (num_features))

        # dictionary, where key is word, value is word vectors
        embedding_model = {}
        for line in open(model_name, 'r'):
            tmp = line.strip().split(" ")
            word, vec = tmp[0], [float(num) for num in tmp[1:]]
            assert(len(vec) == num_features)
            if word not in embedding_model:
                embedding_model[word] = vec
        assert(len(embedding_model) == 400000)

    else:
        raise ValueError('Unknown pretrain model type: %s!' % (model_type))

    embedding_weights = [embedding_model[w] if w in embedding_model
                         else np.random.uniform(-0.25, 0.25, num_features)
                         for w in vocabulary_inv]
    embedding_weights = np.array(embedding_weights).astype('float32')

    return embedding_weights
