import torch
from nltk.tokenize import word_tokenize
import json
import pickle
from EvaluationMethods import *

def padding_mask(seq_k, seq_q):

    # pad sentence
    len_q = seq_q.size(1)
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1,len_q,-1)

    return pad_mask


def sequence_mask(seq):

    batch_size , seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len),dtype = torch.uint8),
                      diagonal = 1)
    mask = mask.unsqueeze(0).expand(batch_size, -1,-1)
    return mask


def build_dictionary(corpus_file):
    # build dictionary from input file

    token2id = {"<eos>": 0,
                "<unk>": 1,
                "<bos>": 2}

    k = 3
    with open(corpus_file) as f:
        for line in f.readlines():
            for token in word_tokenize(line):
                if token not in token2id:
                    token2id[token] = k
                    k += 1

    id2token = dict(zip(token2id.values(),token2id.keys()))

    return token2id, id2token


def print_sentence(x,id2token):
    # print sentence
    print(" ".join([id2token[int(i)] for i in x[0]]))


def evaluation(test_file, image_features, model, id2token, token2id,max_len,device=None):

    # compute the accuracy and wups of the model
    input_y = torch.zeros(1,36,dtype=torch.long)
    input_y[:,0] = 2

    accuracy_of_decoder = 0.0
    wups_of_gen = 0.0
    i = 0
    model = model.to(device)

    with open(test_file) as f:
        corpus = f.readlines()
        num_of_samples = len(corpus) // 2

        for sentence in corpus:
            if i % 2 == 0:
                image = sentence.split(" ")[-2]
                src,tgt = convert_sentence_to_src_and_tgt_id_list(sentence,max_len,token2id)


                input_image = torch.tensor(image_features[image],dtype=torch.float)
                src = src.to(device)
                tgt = tgt.to(device)
                input_image = input_image.view(14*14,512).unsqueeze(0).to(device)
                input_y = input_y.to(device)

                dec_out, gen_out = model(src.unsqueeze(0),tgt.unsqueeze(0),input_image)
                _,predict_out = torch.max(dec_out,dim=2)

                acc = compute_accuracy(src, predict_out.squeeze(0))
                print("{} acc {}".format(i,acc))
                accuracy_of_decoder += acc
            else:
                _,gen_out_ids = torch.max(gen_out,dim=2)
                gen_out = convert_id_list_to_token_list(gen_out_ids.squeeze(0), id2token)

                ground_t = word_tokenize(sentence)
                wups = compute_wups(ground_t, gen_out)
                print("answer:",ground_t)
                print("gen_out",gen_out)
                print(" {} wups {}".format(i,wups))
                wups_of_gen += wups

            i += 1

    return accuracy_of_decoder/num_of_samples, wups_of_gen/num_of_samples


def convert_sentence_to_src_and_tgt_id_list(sentence,max_len,token2id):

    src_list = word_tokenize(sentence)
    tgt_list = ["<bos>"] + word_tokenize(sentence)

    src = torch.zeros(max_len, dtype = torch.long)
    tgt = torch.zeros(max_len, dtype = torch.long)

    src = convert_token_list_to_id_list(src_list, src, token2id)
    tgt = convert_token_list_to_id_list(tgt_list, tgt, token2id)

    return src, tgt


def convert_token_list_to_id_list(token_list, id_list,token2id):

    for i in range(len(token_list)):
        if token_list[i] in token2id:
            id = token2id[token_list[i]]
        else:
            id = 1
        id_list[i] = id
    return id_list


def convert_id_list_to_token_list(id_list,id2token):

    token_list = []
    for id in id_list:
        id = id.item()
        if id == 0:
            break
        if id == 2:
            continue
        token_list.append(id2token[id])
    return token_list


def get_image_feature_from_file(image_file,file_name):

    file = open(file_name,"wb")
    with open(image_file, "r") as f:
        feat = json.load(f)
    pickle.dump(feat,file)
    file.close()


def get_image_feature_from_file_(file_name):

    with open(file_name,"rb") as f:
        features = pickle.load(f)
    return features


