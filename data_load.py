import tensorflow as tf
from utils import calc_num_batches
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

def loadGloVe(filename):
    embd = np.load(filename)
    return embd

def loadGloVe_2(filename, emb_size):
    mu, sigma = 0, 0.1  # 均值与标准差
    rarray = np.random.normal(mu, sigma, emb_size)
    embd = {}
    #embd['<pad>'] = [0]*emb_size
    #embd['<pad>'] = list(rarray)
    embd['<unk>'] = list(rarray)
    file = open(filename,'r')
    for line in tqdm(file.readlines()):
        row = line.rstrip().split(' ')
        if row[0] in embd.keys():
            continue
        else:
            embd[row[0]] = [float(v) for v in row[1:]]
    file.close()
    return embd

def loadw2v(filename, emb_size):
    mu, sigma = 0, 0.1  # 均值与标准差
    rarray = np.random.normal(mu, sigma, emb_size)
    embd = {}
    embd['<pad>'] = [0]*emb_size
    #embd['<pad>'] = list(rarray)
    embd['<unk>'] = list(rarray)
    w2v = Word2Vec.load(filename)
    for word in w2v.wv.vocab:
        if word in embd.keys():
            continue
        else:
            embd[word] = w2v[word]
    return embd

def preprocessVec(gloveFile, vocab_file, outfile):
    emdb = loadGloVe_2(gloveFile, 300)
    trimmd_embd = []
    with open(vocab_file, 'r') as fr:
        for line in fr:
            word = line.rstrip()
            if word in emdb:
                trimmd_embd.append(emdb[word])
            else:
                trimmd_embd.append(emdb['<unk>'])
    np.save(outfile, trimmd_embd)

def preprocess_cn(w2v, vocab_file, outfile):
    embd = loadw2v(w2v, 128)
    trimmd_embd = []
    with open(vocab_file, 'r') as fr:
        for line in fr:
            word = line.rstrip()
            if word in embd:
                trimmd_embd.append(embd[word])
            else:
                trimmd_embd.append(embd['<unk>'])

def load_vocab(vocab_fpath):
    '''Loads vocabulary file and returns idx<->token maps
    vocab_fpath: string. vocabulary file path.
    Note that these are reserved
    0: <pad>, 1: <unk>, 2: <s>, 3: </s>

    Returns
    two dictionaries.
    '''
    with open(vocab_fpath, 'r') as fr:
        vocab = [line.strip() for line in fr]

    # vocab = [line.split()[0] for line in open(vocab_fpath, 'r').read().splitlines()]
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}
    return token2idx, idx2token, len(vocab)

def load_data(fpath, maxlen):
    '''Loads source and target data and filters out too lengthy samples.
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.

    Returns
    sents1: list of source sents
    sents2: list of target sents
    '''
    sents1, sents2 = [], []
    labels = []
    with open(fpath, 'r') as fr:
        for line in fr:
            content = line.strip().split("\t")
            sent1 = content[0].lower()
            sent2 = content[1].lower()
            label = content[2] #cn data
            #label = content[2] #snli data
            if len(sent1.split()) > maxlen:
                #continue
                sent1 = sent1[len(sent1) - maxlen:]#for cn data
            if len(sent2.split()) > maxlen:
                #continue
                sent2 = sent2[len(sent2) - maxlen:]#for cn data
            sents1.append(sent1)
            sents2.append(sent2)
            labels.append([label])
    return sents1, sents2, labels

def removePunc(inputStr):
    string = re.sub(r"\W+", "", inputStr)
    return string.strip()

def encode(inp, dict, maxlen):
    '''Converts string to number. Used for `generator_fn`.
    inp: 1d byte array.
    type: "x" (source side) or "y" (target side)
    dict: token2idx dictionary

    Returns
    list of numbers
    '''
    #for cn dataset
    x = [dict.get(t, dict["<unk>"]) for t in inp]
    #for snli dateset
    # x = []
    # for i in re.split(r"\W+", inp):
    #     i = i.strip()
    #     i = removePunc(i)
    #     i = i.lower()
    #     if i == "":
    #         continue
    #     x.append(dict.get(i, dict["<unk>"]))
    x = pad_sequences([x], maxlen=maxlen, dtype='int32',padding='post')
    #print(x)
    #x = [dict.get(t, dict["<unk>"]) for t in re.split(r"\W+'", inp)]
    return x[0]

def generator_fn(sents1, sents2, labels, maxlen, vocab_fpath):
    '''Generates training / evaluation data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.
    yields
    xs: tuple of
        x: list of source token ids in a sent
        x_seqlen: int. sequence length of x
        sent1: str. raw source (=input) sentence
    labels: tuple of
        decoder_input: decoder_input: list of encoded decoder inputs
        y: list of target token ids in a sent
        y_seqlen: int. sequence length of y
        sent2: str. target sentence
    '''
    token2idx, _, _ = load_vocab(vocab_fpath)
    enc = OneHotEncoder(sparse=False, categories='auto')
    labelList = enc.fit_transform(labels)
    for sent1, sent2, label in zip(sents1, sents2, labelList):
        x = encode(sent1.decode(), token2idx, maxlen)
        y = encode(sent2.decode(), token2idx, maxlen)
        x_seqlen, y_seqlen = len(x), len(y)

        yield (x, y, x_seqlen, y_seqlen, label)


def generator_fn_infer(sents1, sents2, maxlen, vocab_fpath):
    token2idx, _, _ = load_vocab(vocab_fpath)
    for sent1, sent2 in zip(sents1, sents2):
        x = encode(sent1.decode(), token2idx, maxlen)
        y = encode(sent2.decode(), token2idx, maxlen)
        x_seqlen, y_seqlen = len(x), len(y)
        yield (x, y, x_seqlen, y_seqlen)


def input_fn(sents1, sents2, labels, maxlen, vocab_fpath, batch_size, shuffle=False):
    '''Batchify data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean
    Returns
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    '''
    # ((x, x_seqlen), (y, y_seqlen), (label))
    shapes = ([None], [None], (), (), [None])
    types = (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32)
    paddings = (0, 0, 0, 0, 0)


    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(sents1, sents2, labels, maxlen, vocab_fpath))  # <- arguments for generator_fn. converted to np string arrays

    if shuffle:  # for training
        dataset = dataset.shuffle(128 * batch_size)

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

    return dataset


def input_fn_infer(sents1, sents2, maxlen, vocab_fpath, batch_size):
    # ((x, x_seqlen), (y, y_seqlen))
    shapes = ([None], [None], (), ())
    types = (tf.int32, tf.int32, tf.int32, tf.int32)
    paddings = (0, 0, 0, 0)

    dataset = tf.data.Dataset.from_generator(
        generator_fn_infer,
        output_shapes=shapes,
        output_types=types,
        args=(sents1, sents2, maxlen, vocab_fpath))  # <- arguments for generator_fn. converted to np string arrays

    #dataset = dataset.repeat()  # iterate forever
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

    return dataset


def get_batch(fpath, maxlen, vocab_fpath, batch_size, shuffle=False):
    '''Gets training / evaluation mini-batches
    fpath: source file path. string.
    maxlen: target sent maximum length. scalar.
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean
    Returns
    batches
    num_batches: number of mini-batches
    num_samples
    '''
    sents1, sents2, labels = load_data(fpath, maxlen)
    batches = input_fn(sents1, sents2, labels, maxlen, vocab_fpath, batch_size, shuffle=shuffle)
    num_batches = calc_num_batches(len(sents1), batch_size)

    return batches, num_batches, len(sents1)


def get_batch_infer(sents1, sents2, maxlen, vocab_fpath, batch_size):
    batches = input_fn_infer(sents1, sents2, maxlen, vocab_fpath, batch_size)
    return batches

if __name__ == '__main__':
    preprocessVec("./data/vec/glove.840B.300d.txt", "./data/snli.vocab", "./data/vec/snil_trimmed_vec.npy")
