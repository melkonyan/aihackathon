import os
import json
import pickle
import urllib2
import nltk
import numpy as np

def prepare_news_datasets(max_vocab_size, max_samples_num):
    signal_news_file = 'signal.jsonl'
    signal_contetnt_fiel = 'signal.txt'
    if not os.path.exists("new_data/"):
        os.makedirs("new_data/")
    if not os.path.exists('news_data/signal.jsonl'):
        #fname = 'signalmedia-1m.jsonl.gz'
        fname = downloadFile('http://research.signalmedia.co/newsir16/signalmedia-1m.jsonl.gz')
        import gzip
        import shutil
        with gzip.open(fname, 'rb') as f_in, open('news_data/signal.jsonl', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    if not os.path.exists('news_data/signal.txt') or True:
        with open('news_data/signal.jsonl', 'r') as fin, open('news_data/signal.txt', 'w') as fout:
            for i in range(max_samples_num):
                sample = json.loads(fin.readline())
                title = sample['title'].replace('\n', '').encode('ascii',errors='ignore')
                content = sample['content'].replace('\n', '').encode('ascii',errors='ignore')
                text =  title + '. ' + content
                fout.write(text)
    return
    if os.path.exists("news_data/vocab.txt"):
        print "vocab mapping found..."
    else:
        print "no vocab mapping found, running preprocessor..."
        createVocab([], max_vocab_size)
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed/")
        print "No processed data file found, running preprocessor..."
    else:
        return
    import vocabmapping


#method from:
#http://stackoverflow.com/questions/22676/how-do-i-download-a-file-over-http-using-python
def downloadFile(url):
    file_name = os.path.join("data/", url.split('/')[-1])
    u = urllib2.urlopen(url)
    f = open(file_name, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "Downloading: %s Bytes: %s" % (file_name, file_size)
    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break
        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print status,
    f.close()
    return file_name


'''
This function tokenizes sentences
'''
def tokenize(text):
    text = text.decode('utf-8')
    return nltk.word_tokenize(text)


'''
create vocab mapping file
'''
def createVocab(files, max_vocab_size, max_samples_num):
    print "Creating vocab mapping..."
    dic = {}
    for f in files:
        with open(f, 'r') as review:
            tokens = tokenize(review.read().lower())
            for t in tokens:
                if t not in dic:
                    dic[t] = 1
                else:
                    dic[t] += 1
    d = {}
    counter = 0
    for w in sorted(dic, key=dic.get, reverse=True):
        d[w] = counter
        counter += 1
        #take most frequent 50k tokens
        if counter >=max_vocab_size:
            break
    #add out of vocab token and pad token
    d["<UNK>"] = counter
    counter +=1
    d["<PAD>"] = counter
    with open('news_data/vocab.txt', 'wb') as handle:
        pickle.dump(d, handle)


def convert_words_to_vec(fname, vocab_mapping, max_seq_length, is_fake):
   data = np.array([i for i in range(max_seq_length + 2)])
   with open(fname, 'r') as review:
        tokens = tokenize(review.read().lower())
        numTokens = len(tokens)
        indices = [vocab_mapping.getIndex(j) for j in tokens]
        #pad sequence to max length
        if len(indices) < max_seq_length:
            indices = indices + [vocab_mapping.getIndex("<PAD>") for i in range(max_seq_length - len(indices))]
        else:
            indices = indices[0:max_seq_length]
        if  is_fake:
            indices.append(1)
        else:
            indices.append(0)
        indices.append(min(numTokens, max_seq_length))
        assert len(indices) == max_seq_length + 2, str(len(indices))
        data = np.vstack((data, indices))
        indices = []
        #remove first placeholder value
        data = data[1::]
        saveData(data)


'''
Saves processed data numpy array
'''
def saveData(npArray, fname):
    name = "{}.npy".format(fname)
    outfile = os.path.join("data/processed/", name)
    print "numpy array is: {0}x{1}".format(len(npArray), len(npArray[0]))
    np.save(outfile, npArray)

if __name__ == '__main__':
    prepare_news_datasets(100, 50000)