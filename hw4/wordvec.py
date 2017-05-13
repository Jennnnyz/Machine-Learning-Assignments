import numpy as np
import word2vec
import nltk
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from adjustText import adjust_text

train = True
k = 1000

def training():
    # DEFINE your parameters for training
    MIN_COUNT = 0
    WORDVEC_DIM = 300
    WINDOW = 5
    NEGATIVE_SAMPLES = 5
    ITERATIONS = 6
    MODEL = 0
    LEARNING_RATE = 0.025

    word2vec.word2vec(train = 'all-phrases',
        output = 'all.bin', 
        cbow = MODEL,
        size = WORDVEC_DIM,
        min_count = MIN_COUNT,
        window = WINDOW,
        negative = NEGATIVE_SAMPLES,
        iter_ = ITERATIONS,
        alpha = LEARNING_RATE,
        verbose = True)



def prediction():
    model = word2vec.load('all.bin')
    vocabs = []
    vecs = []
    for vocab in model.vocab:
        vocabs.append(vocab)
        vecs.append(model[vocab])
    vecs = np.array(vecs)[:k]
    vocabs = vocabs[:k]

    #Dimensionality Reduction

    tsne = TSNE(n_components = 2)
    reduced = tsne.fit_transform(vecs)

    # filtering
    use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
    puncts = ["'", '.', ':', ";", ',', "?", "!", u"’"]

    plt.figure()
    texts = []
    for i, label in enumerate(vocabs):
        pos = nltk.pos_tag([label])
        if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tags
                and all(c not in label for c in puncts)):
            x, y = reduced[i, :]
            texts.append(plt.text(x, y, label))
            plt.scatter(x, y)

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))

    plt.show()


def phrases(model,word):
    indexes, metrics = model.cosine('Harry')
    print(model.generate_response(indexes, metrics).tolist())

def analogies(model,pos, neg):
    indexes, metrics = model.analogy(pos=pos, neg=neg, n=10)
    print(model.generate_response(indexes, metrics).tolist())

def main():
    if train:
        training()
    prediction()

if __name__ == "__main__":
    main()
