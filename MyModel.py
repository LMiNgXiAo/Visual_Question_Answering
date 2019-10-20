from DataGenerator import DataGenerator
from torch.utils.data import DataLoader
from Transformer import Transformer
import numpy as np
from torch import optim
from torch.nn import CrossEntropyLoss
from Utils import *

class Model(object):

    def __init__(self,model,parameters, id2token,token2id):

        self.model = model
        self.parameters = parameters
        self.id2token = id2token
        self.token2id = token2id

    def train(self,dataloader,image_features):

        device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

        optimizer = optim.Adam(self.model.parameters(),lr=self.parameters["learning_rate"])
        criterion = CrossEntropyLoss()

        for epoch in range(1,self.parameters["epoches"]):

            print("Epoch {}.".format(epoch))
            print("_"*50)

            running_loss = 0

            for i,(src_x,tgt_x,src_y,tgt_y,images) in enumerate(dataloader):

                input_image = torch.cat([torch.tensor(np.array(image_features[image]), \
                                                      dtype=torch.float).view(14*14,512).unsqueeze(0) \
                                         for image in images],dim=0)
                input_image = input_image.to(device)
                src_x = src_x.to(device)
                tgt_x = tgt_x.to(device)
                src_y = src_y.to(device)
                tgt_y = tgt_y.to(device)

                self.model = self.model.to(device)

                dec_output, gen_output = self.model(src_x, tgt_x,input_image)
                _, ids = torch.max(gen_output, dim=2)

                loss = criterion(dec_output.permute(0,2,1), src_x)  + \
                       criterion(gen_output.permute(0,2,1), src_y)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i % 10 == 0 and i != 0:
                    print("The average loss: {:.3f}".format(running_loss/10))
                    running_loss = 0
        print("Training Comleted")
        torch.save(self.model.state_dict(),"./transformer")
        print("The model has been saved successfully")


    def reproduce_question(self, input,img_features):

        img_in = input.split(" ")[-2]
        input_image = torch.tensor(np.array(img_features[img_in]), \
                                              dtype=torch.float).view(14 * 14, 512).unsqueeze(0)

        src_x, tgt_x = convert_sentence_to_src_and_tgt_id_list(input,self.parameters["max_len"],self.token2id)

        dec_out,gen_out = self.model(src_x.unsqueeze(0), tgt_x.unsqueeze(0), input_image)
        _, p = torch.max(dec_out, dim= 2)

        return " ".join(convert_id_list_to_token_list(p[0],self.id2token))


    def answer_question(self, input,img_features):

        src_x, tgt_x = convert_sentence_to_src_and_tgt_id_list(input, self.parameters["max_len"], self.token2id)

        img_in = input.split(" ")[-2]
        input_image = torch.tensor(np.array(img_features[img_in]), \
                                   dtype=torch.float).view(14 * 14, 512).unsqueeze(0)

        dec_out, gen_out = self.model(src_x.unsqueeze(0), tgt_x.unsqueeze(0), input_image)
        _,ids =torch.max(gen_out,dim=2)

        return " ".join(convert_id_list_to_token_list(ids[0],self.id2token))








