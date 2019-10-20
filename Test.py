from Utils import *
from Transformer import Transformer
from torch.utils.data import DataLoader
from DataGenerator import DataGenerator
from MyModel import Model

token2id, id2token = build_dictionary("qa.894-raw-train.txt")

def print_sentence(id2token,x):
    print(" ".join([id2token[int(i)] for i in x[0]]))

parameters ={"vocab_size": len(token2id),
                 "max_len" : 36,
                 "num_layers":6,
                 "d_model": 512,
                 "num_heads": 8,
                 "ffn_dim":1024,
                 "dropout":0.2,
                 "learning_rate":1e-4,
                 "batch_size":1,
                 "epoches": 30}

#dataset = DataGenerator("qa.37-reduced-test.txt",parameters["max_len"],token2id)
#dataloader = DataLoader(dataset, batch_size=parameters["batch_size"], num_workers=2)
image_features = get_image_feature_from_file_("image_features")

device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

model = Transformer(vocab_size = parameters["vocab_size"],
                              max_len = parameters["max_len"],
                              num_layers = parameters["num_layers"],
                              d_model = parameters["d_model"],
                              num_heads = parameters["num_heads"],
                              ffn_dim = parameters["ffn_dim"],
                              dropout = parameters["dropout"],
                              device = device,
                    )
#model = Model(Transformer,parameters,id2token,token2id)
#model.train(dataloader,image_features)

model.load_state_dict(torch.load("transformer-5.dms",map_location='cpu'))

model = Model(model,parameters,id2token,token2id)
print(model.answer_question("what is the object next to the bed in the image526 ?",image_features))

#acc,wups = evaluation("qa.37-reduced-test.txt",image_features,model,
      #                id2token,token2id,parameters["max_len"])
#print(acc)
#print(wups)
