import config
from random import randint

from config import BUCKETS

'''
author: lee chung
description: this is a project for computer linguitics course.
           i used cells from https://github.com/NickShahML/tensorflow_with_latest_papers, many thanks to those hardcore coders
'''

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# default tokenizer for sentence
def tokenize(sen):
   return sen.lower().strip().split()


# transform sentence to id sequence from vocab id
def sentence2tokenids(sen, vocab , tokenize=tokenize ):
   words = tokenize(sen)
   return [vocab.get(w,UNK_ID) for w in words]

# transform id sequence back to sentence using reverse vocab 
def tokenid2sentence(tid , vocab):
   return(vocab[i] for i in tid)


def create_vocab( data_path , vocab_path , max_vocab_size = 10000 , tokenize=tokenize  ):
   '''
      create vocabulary from training data, and store vocab in vocab_path
   '''
   print("Creating vocabulary")
   vocab = {}
   with open(data_path,'r') as f:
      for line in f:
         t = line.strip().split('\t')
         #print t
         if len(t) is 2:
            for to in tokenize(t[0]):
               if to in vocab:
                  vocab[to] += 1
               else:
                  vocab[to] = 1
            for to in tokenize(t[1]):
               if to in vocab:
                  vocab[to] += 1
               else:
                  vocab[to] = 1
   vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
   #print(len(vocab_list))
   if len(vocab_list) > max_vocab_size:
      vocab_list = vocab_list[:max_vocab_size]
   f = open(vocab_path,'w')
   for i in vocab_list:
      f.write( i + '\n' )


def init_vocab(vocab_path):
   '''
      initialize vocabulary using vocab file, returns a vocab list and re_vocab of id,vocab
   '''
   re_vocab = []
   with open(vocab_path) as f:
      re_vocab.extend(f.read().splitlines())
   #print re_vocab
   vocab = dict([(x,y) for (y,x) in enumerate(re_vocab)])
   #print vocab['_PAD']
   return vocab, re_vocab
   

def get_train_data(data_path , vocab):
   '''
      create training set from data_file using the vocab list
   '''
   train = [[] for _ in config.BUCKETS]
   print('Creating training data')
   with open(data_path) as f:
      for line in f:
         t = line.strip().split('\t')
         if len(t) is 2:
            #train.append([sentence2tokenids(t[0],vocab),sentence2tokenids(t[1],vocab)])
            source_ids = sentence2tokenids(t[0],vocab)
            target_ids = sentence2tokenids(t[1],vocab)
            target_ids.append(EOS_ID)
            for bucket_id, (source_size ,target_size) in enumerate(config.BUCKETS):
               if len(source_ids) < source_size and len(target_ids) < target_size:
                  train[bucket_id].append([source_ids,target_ids])
                  break
   return train




def test():
   # create vocab using data set and store vocab in vocab path\
   create_vocab(config.FLAGS.data_set , config.FLAGS.vocab )
   # init vocab and reverse vocab
   vocab , vocab_re =  init_vocab(config.FLAGS.vocab)
   # get training set 
   train_data = get_train_data(config.FLAGS.data_set,vocab)
   return train_data
   

if __name__ == '__main__':
   print test()
