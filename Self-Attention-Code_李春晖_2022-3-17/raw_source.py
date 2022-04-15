import numpy as np
from tqdm import tqdm
import pickle as pkl
import os

class Preload_Data:

    def __init__(self,config):
        self.UNK,self.PAD = '<UNK>', '<PAD>'
        self.MAX_VOCAB_SIZE = 10000
        self.config = config

    def build_vocab(self,file_path, tokenizer, max_size, min_freq):

        vocab_dic = {}
        with open(file_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content = lin.split('\t')[0]
                for word in tokenizer(content):
                    vocab_dic[word] = vocab_dic.get(word, 0) + 1
            vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                         :max_size]
            vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
            vocab_dic.update({self.UNK: len(vocab_dic), self.PAD: len(vocab_dic) + 1})
        return vocab_dic

    def build_dataset(self):
        tokenizer = lambda x: [y for y in x]
        if os.path.exists(self.config.vocab_path):
            vocab = pkl.load(open(self.config.vocab_path, 'rb'))
        else:
            vocab = self.build_vocab(self.config.train_path, tokenizer=tokenizer, max_size=self.MAX_VOCAB_SIZE, min_freq=1)
            pkl.dump(vocab, open(self.config.vocab_path, 'wb'))
        vocab_size = len(vocab)
        def load_dataset(path,):
            all_contents = []
            labels = []
            # seq_lens = []
            max_len = 0
            with open(path, 'r', encoding='UTF-8') as f:
                
                for line in tqdm(f):
                    lin = line.strip()
                    if not lin:
                        continue
                    temp_list = lin.split('\t')
                    # print(temp_list)
                    label = temp_list[1]
                    if len(temp_list) > 2:

                        content = ''
                        for i in range(1, len(temp_list)):
                            content += temp_list[i]
                        # print('content:',content)
                    elif len(temp_list) == 2:
                        content = temp_list[0]
                        # print(content)
                    else:
                        raise ValueError
                    if len(content) > max_len:
                        max_len = len(content)
                    all_contents.append(content)
                    labels.append(label)
                    
            res_sentences = []
            res_labels = []
            res_masks = []
            pad_size = max_len
            for i in range(len(all_contents)):
                content = all_contents[i]
                label = labels[i]
                mask = np.zeros(max_len)
                
                mask[:len(content)] = 1
                words_line = []
                token = tokenizer(content)
                # seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([self.PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size

                for word in token:
                    words_line.append(vocab.get(word, vocab.get(self.UNK)))
                res_sentences.append(words_line)
                res_labels.append(int(label))
                res_masks.append(mask)
                # seq_lens.append(seq_len)
            return res_sentences,res_labels,res_masks,all_contents

        train_text,train_labels,train_masks,train_contents = load_dataset(self.config.train_path)
        train_masks = np.array(train_masks)
        dev_text,dev_labels,dev_masks,dev_contents = load_dataset(self.config.dev_path)
        dev_masks = np.array(dev_masks)
        test_text,test_labels,test_masks,test_contents = load_dataset(self.config.test_path)
        test_masks = np.array(test_masks)

        return train_text,train_labels,train_masks,dev_text,dev_labels,dev_masks,\
    test_text,test_labels,test_masks,vocab_size,train_contents,dev_contents,test_contents

class Dataset:
    def __init__(self,text,labels,masks):
        self.text = torch.tensor(text)
        self.labels = torch.tensor(labels)
        self.masks = torch.from_numpy(masks)
    def __getitem__(self, index):
        return self.text[index],self.labels[index],self.masks[index]
    def __len__(self):
        return len(self.text)


class arg_parser:
    def __init__(self):
        self.train_path = r'data\train.txt'
        self.dev_path = r'data\dev.txt'
        self.test_path = r'data\test.txt'
        self.vocab_path = r'data\vocab.pkl'
import torch

def main():
	args = arg_parser()
	p = Preload_Data(args)
	train_text,train_labels,train_masks,dev_text,dev_labels,dev_masks,test_text,test_labels,test_masks,vocab_size,train_contents,dev_contents,test_contents = p.build_dataset()
	return train_text,train_labels,train_masks,dev_text,dev_labels,dev_masks,test_text,test_labels,test_masks,vocab_size,train_contents,dev_contents,test_contents
