from torch.utils.data import DataLoader, Dataset
from Utils import *


class DataGenerator(Dataset):

    def __init__(self,datafile,max_seq_len, token2id):

        self.token2id = token2id
        self.max_seq_len = max_seq_len
        self.src_x, self.tgt_x, self.src_y, self.tgt_y,self.images = self.get_data_from_file(datafile)


    def __getitem__(self, index):
        return self.src_x[index], self.tgt_x[index], self.src_y[index], self.tgt_y[index],self.images[index]


    def __len__(self):
        return len(self.tgt_y)


    def convert_tokenlist2idlist(self,token_list,id_list):

        for i in range(len(token_list)):
            if token_list[i] in self.token2id:
                id = self.token2id[token_list[i]]
            else:
                id = 1
            id_list[i] = id
        return id_list


    def get_data_from_file(self,datafile):
        # read data from file
        k = 0
        tgt_x = []
        src_x = []
        tgt_y = []
        src_y = []
        images = []

        with open(datafile) as f:

            for line in f.readlines():

                src_sentence, tgt_sentence = convert_sentence_to_src_and_tgt_id_list(line,
                                                                      self.max_seq_len,
                                                                      self.token2id)
                if k % 2 == 0:
                    images.append(line.split(" ")[-2])
                    src_x.append(src_sentence)
                    tgt_x.append(tgt_sentence)
                else:
                    src_y.append(src_sentence)
                    tgt_y.append(tgt_sentence)#

                k += 1

        return src_x,tgt_x,src_y,tgt_y,images


