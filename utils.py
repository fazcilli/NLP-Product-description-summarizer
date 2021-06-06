import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import time
import math

from preprocess.clean import clean_contractions
from variables import *

def indexesFromSentence(vocab, sentence):
    return [vocab.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(vocab, sentence):
    indexes = indexesFromSentence(vocab, sentence)
    indexes.append(EOS_TOKEN)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(vocab, pair):
    input_tensor = tensorFromSentence(vocab, pair[0])
    target_tensor = tensorFromSentence(vocab, pair[1])
    return (input_tensor, target_tensor)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()


def word_count_bins():
    data = pd.read_csv('./data/all.csv', usecols=['Long Description', 'Short Description'])
    ld_count = []
    sd_count = []
    cv = CountVectorizer()
    for i, row in data.iterrows():
        try:
            cv.fit([row[0]])
            ld_count.append(len(cv.vocabulary_))
        except:
            ld_count.append(0)
        try:
            cv.fit([row[1]])
            sd_count.append(len(cv.vocabulary_))
        except:
            sd_count.append(0)
        # ld_count.append(len(row[0].split(' ')))
        # sd_count.append(len(row[1].split(' ')))
    print(np.max(ld_count))
    print(np.max(sd_count))

    length_df = pd.DataFrame({'long desc': ld_count, 'short desc': sd_count})
    length_df.hist(bins=15, figsize=(16, 8))
    plt.show()

def overlap_count_bins():
    data = pd.read_csv('./data/all.csv', usecols=['Long Description', 'Short Description'])
    ld_count = []
    for i, row in data.iterrows():
        l = len(set(clean_contractions(row[0]).split(' ')).intersection(set(clean_contractions(row[1]).split(' '))))
        ld_count.append(min(80, l))
    length_df = pd.DataFrame({'': ld_count})
    length_df.hist(bins=20, figsize=(10, 8))
    plt.show()

def load_data():
    total = 0
    df = pd.DataFrame()
    u = 40
    for i in range(u):
        data = pd.read_csv('./data/products%d.csv' % i, index_col='Id')
        total += data.shape[0]
        df = df.append(data, ignore_index=True)
    df.to_csv('./data/all.csv')
    print('Number of products: ', total)

def split_data():
    data = pd.read_csv('./data/all_cleaned.csv')
    (n, _) = data.shape
    test = data.sample(n=1000, random_state=1)
    test.drop(test.columns[[0, 1]], axis=1, inplace=True)
    test.to_csv('./data/test.csv')
    print(test.shape)
    data = data.drop(test.index)
    validation = data.sample(n=1000, random_state=1)
    validation.drop(validation.columns[[0, 1]], axis=1, inplace=True)
    validation.to_csv('./data/val.csv')
    print(validation.shape)
    data = data.drop(validation.index)
    train = data.sample(n=5000, random_state=1)
    train.drop(train.columns[[0, 1]], axis=1, inplace=True)
    train.to_csv('./data/train.csv')
    print(train.shape)

if __name__ == '__main__':
    overlap_count_bins()
    # split_data()
