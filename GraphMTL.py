#!/usr/bin/env python
import torch.nn as nn
import torch
from pytorch_pretrained_bert import BertModel
from torch.autograd import Variable
import torch.nn.functional as F
import random
import numpy as np
class Sent_NN(nn.Module):
    """docstring for Sent_NN"""
    def __init__(self, args, output_dim=7):
        super(Sent_NN, self).__init__()
        self.args = args
        self.output_dim = output_dim 
        self.fc = nn.Linear(768, self.output_dim)
        nn.init.xavier_normal_(self.fc.weight)
        self.sigmoid = nn.Sigmoid()  
    def forward(self, hidden):
        # out =  hidden
        out = self.fc(hidden)
        if self.output_dim == 2:
            out = self.sigmoid(out)
        else:
            # out = F.softmax(out, dim=1)
            out = out##self.sigmoid(out)
        return out

        
class PDN_NN(nn.Module):
    """docstring for PDN_NN"""
    def __init__(self, args, output_dim=5):
        super(PDN_NN, self).__init__()
        self.args = args

        self.fc = nn.Linear(768, output_dim)
        self.sigmoid = nn.Sigmoid() 
    def forward(self, hidden):
        out = self.fc(hidden)
        pose = self.sigmoid(out)

        return out, pose 
class BERT_base(nn.Module):
  
    def __init__(self, args, device, vocab_size, pre_embedding):
        super(BERT_base, self).__init__()
        self.num_labels = args.class_dim
        self.n_layers = args.n_layers
        self.hidden_dim = 768
        self.device =  device
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(args.drop_prob) 
        hidden_dim_bert = self.bert.config.to_dict()['hidden_size']
        self.rnn = nn.GRU(hidden_dim_bert,
                          self.hidden_dim,
                          num_layers = self.n_layers,
                          bidirectional = args.bidirectional,
                          batch_first = True,
                          dropout = 0 if self.n_layers < 2 else args.dropout)
        # self.out = nn.Linear(self.hidden_dim * 2 if args.bidirectional else self.hidden_dim, 768)
        self.out = nn.Linear(self.hidden_dim, 768)

    def forward(self, input_ids, hidden, token_type_ids=None, attention_mask=None, labels=None):
        # with torch.no_grad():
        bert_out, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # _, hidden = self.rnn(bert_out)
        # if self.rnn.bidirectional:
        #     hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        # else:
        #     hidden = self.dropout(hidden[-1,:,:])
        pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        # print(hidden.size())
        # pooled_output = self.out(pooled_output) 
        return pooled_output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden
class MTL_Input(nn.Module):
    """docstring for MTL_Input"""
    def __init__(self, args, device, vocab_size, pre_embedding):
        super(MTL_Input, self).__init__()
        self.args = args
        self.embedding_dim = args.embedding_dim

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.embedding.weight = nn.Parameter(torch.from_numpy(pre_embedding), requires_grad=True) 

        # self.bert = BertModel.from_pretrained('bert-base-uncased')



    def forward(self, x):
        x = x.long()
        # embedded, pooled_output = self.bert(x, None, None, output_all_encoded_layers=False)


        embedded = self.embedding(x)

        return embedded

class SiLGMTL(nn.Module):
    def __init__(self, args, device):
        super(Gated_MTL, self).__init__()

        self.embedding_dim = args.embedding_dim
        n_filters = 256
        filter_sizes = [3, 4, 5]
        output_dim = 768
        dropout = 0.5
        self.convs1 = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, self.embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        self.convs2 = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, self.embedding_dim)) 
                                    for fs in filter_sizes
                                    ])        
        self.fc = nn.Linear(output_dim*2, output_dim)
        self.gate1 = nn.Linear(len(filter_sizes) * n_filters , output_dim)
        self.gate2 = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()  

        self.fcPDN = nn.Linear(768, 5)

        self.fcSent = nn.Linear(768, 7)
        self.cosine_1 = nn.CosineSimilarity(dim=1)
        self.cosine_2 = nn.CosineSimilarity(dim=2)

        # self.att1 = nn.Linear(300, output_dim)
        # self.att2 = nn.Linear(300, output_dim)
        self.transformer = nn.Transformer(d_model=self.embedding_dim, nhead=15, num_encoder_layers=2)

 


    def forward(self, x1, x2):
        ## x1 batch size, sent len, emb dim
        # x1_att1 = self.att1(x1)
        # x2_att1 = self.att2(x2)
        # x1_att = (F.softmax(x1_att1, dim=1)*x1_att1).sum(dim=1)
        # # x2_att = (F.softmax(x2_att1, dim=2)*x2_att1).sum(dim=1)## batch side x out
        # x1 = x1.permute(1,0,2)
        # x2 = x2.permute(1,0,2)
        # # # print(x1.size(),x2.size())

        # x1 = self.transformer(x1, x2)
        # x2 = self.transformer(x2, x1)
        # x1 = x1.permute(1,0,2)
        # x2 = x2.permute(1,0,2)

        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)

        # x = torch.cat([x1, x2], dim =1)
        # print(x.size(), x1.size())

        # x = x.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        conved1 = [F.relu(conv(x1)).squeeze(3) for conv in self.convs1]

        conved2 = [F.relu(conv(x2)).squeeze(3) for conv in self.convs2]


        convedTo1 = [self.sigmoid(conv)*conv  for conv in conved2]
        convedTo2 = [self.sigmoid(conv)*conv  for conv in conved1]

        conved1 = [conved1[i] + convedTo1[i] for i in range(len(conved1))]
        conved2 = [conved2[i] + convedTo2[i] for i in range(len(conved2))]

        cosine_value1 = [self.cosine_2(conved1[i], conved2[i]).mean(dim=1, keepdim=True) for i in range(len(conved1))]
        # print(cosine_value1[0].size())
        
        self.cosine_value1 = torch.cat(cosine_value1, dim=1).mean(dim=1)
        # print(cosine_value1.size())
        # conved2 = [F.relu(conv(x2)).squeeze(3) for conv in self.convs2]
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        pooled1 = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved1]
        #pooled_n = [batch size, n_filters]
        pooled2 = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved2]

        convedTo1 = [self.sigmoid(conv)*conv for conv in pooled2]
        convedTo2 = [self.sigmoid(conv)*conv for conv in pooled1]

        pooled1 = [pooled1[i] + convedTo1[i] for i in range(len(pooled1))]
        pooled2 = [pooled2[i] + convedTo2[i] for i in range(len(pooled2))]

        out1 = torch.cat(pooled1, dim = 1)
        out2 = torch.cat(pooled2, dim = 1) 
        self.cosine_value2 = self.cosine_1(out1, out2)

        out1 = self.dropout(out1)
        out2 = self.dropout(out2)

      
        att1 = self.sigmoid(out1)*out1
 
        att2 = self.sigmoid(out2)*out2
        # att2 = F.softmax(out2, dim=1)
        # att2_1 = torch.exp(out2)
        # att2 = att2_1/ (1e-8 + (att2_1.sum(dim=1, keepdim=True)))

        out1 = out1+att2#*x1_att
        out2 = out2+att1#*x2_att
        self.cosine_value3 = self.cosine_1(out1, out2)
        # out = torch.cat([out1, out2], dim=1)
        # out = out1 + out2

        out_sent = self.fcSent(out1)
        out_PDN = self.fcPDN(out2)
        out_PDN_pose = self.sigmoid(out_PDN)
        # out = (self.fc(out))

        return out_sent, out_PDN, out_PDN_pose

class CAGMTL(nn.Module):
    def __init__(self, args, device):
        super(CA_MTL, self).__init__()

        self.embedding_dim = args.embedding_dim
        n_filters = 256
        filter_sizes = [3, 4, 5]
        output_dim = 768
        dropout = 0.5
        self.convs1 = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, self.embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        self.convs2 = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, self.embedding_dim)) 
                                    for fs in filter_sizes
                                    ])        
        self.fc = nn.Linear(output_dim*2, output_dim)
        self.gate1 = nn.Linear(len(filter_sizes) * n_filters , output_dim)
        self.gate2 = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()  

        self.fcPDN = nn.Linear(768, 5)

        self.fcSent = nn.Linear(768, 7) 
        self.cosine_1 = nn.CosineSimilarity(dim=1)
        self.cosine_2 = nn.CosineSimilarity(dim=2)

        # self.att1 = nn.Linear(300, output_dim)
        # self.att2 = nn.Linear(300, output_dim)
        # self.transformer = nn.Transformer(d_model=self.embedding_dim, nhead=15, num_encoder_layers=2)

    def get_adj(self, x):
        x = x.permute(1,0,2)
        x_len = x.size()[0]
        adj = torch.zeros(x_len, x_len)
        for k in range(x_len):
            x_i = x[k]
            for kj in range(x_len):
                x_j = x[kj]

    def get_crosssim(self, x1, x2, dim=1):
        lenx = len(x1) 
      
        S = []

        for i in range(lenx):

            x1_i = x1[i]  ## batch_size x out_channel x seq_len(dynamic)
            # print(x1_i.size())
            x2_i = x2[i]
            sij = self.cosine_1(x1_i, x2_i).unsqueeze(dim=1)
            # print(sij.size())
            S.append(sij) ##batch_size x 1 x seq_len

                    
               
                # S[:,j,i] = sij
        return S

    def get_crossatten(self, x1, x2, W):
        atten_size = x1.size()[2]

        # W = nn.Parameter(torch.FloatTensor(atten_size, atten_size)).cuda()
        # torch.nn.init.xavier_uniform_(W)

        out = torch.matmul(x1, W) ## batch_size x out_channel x seq_len
        x2 = x2.permute(0, 2, 1)
        out = torch.bmm(out, x2)## batch_size x out_channel x out_channel
        out = F.softmax(out, dim = 2) 
        return out

    def get_parameters(self, x, dim=1):
        paras = []
        for i in range(len(x)):
            atten_size = x[i].size()[dim]

            self.W = nn.Parameter(torch.FloatTensor(atten_size, atten_size)).cuda()
            torch.nn.init.xavier_uniform_(self.W)
            paras.append(self.W)
        return paras

    def get_crossatten2(self, x1, x2, W):
        # print(x1.size())
        atten_size = x1.size()[1]

        # W = nn.Parameter(torch.FloatTensor(atten_size, atten_size)).cuda()
        # torch.nn.init.xavier_uniform_(W)

        out = torch.mm(x1, W) ## batch_size x out_channel x seq_len
        x2 = x2.t()
        out = torch.mm(out, x2)## batch_size x out_channel x out_channel
        out = F.softmax(out, dim = 1) 
        return out
    def forward(self, x1, x2):
        ## x1 batch size, sent len, emb dim
 

        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)

        # x = torch.cat([x1, x2], dim =1)
        # print(x.size(), x1.size())

        # x = x.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        conved1 = [F.relu(conv(x1)).squeeze(3) for conv in self.convs1]
        conved2 = [F.relu(conv(x2)).squeeze(3) for conv in self.convs2]
        # S = self.get_crosssim(conved1, conved2) #batchsize x len(conved2)
        paras = self.get_parameters(conved1, dim=2)
        attens1 = [self.get_crossatten(conved1[i], conved2[i], paras[i]) for i in range(len(conved2))]
        attens2 = [self.get_crossatten(conved2[i], conved1[i], paras[i]) for i in range(len(conved2))]

        # conved1 = [conved1[i] + self.softmax(S[i], dim=2) for i in range(len(conved1))]
        # conved2 = [conved2[i]*S_1[:,i,:] for i in range(len(conved2))]

        convedTo1 = [torch.bmm(attens2[i], conved2[i]) for i in range(len(conved2))]
        convedTo2 = [torch.bmm(attens1[i], conved1[i]) for i in range(len(conved1))]



        # convedTo1 = [F.softmax(S[i], dim=2)*conved2[i]  for i in range(len(conved2))]
        # convedTo2 = [F.softmax(S[i], dim=2)*conved1[i]  for i in range(len(conved2))]

        conved1 = [conved1[i] + convedTo1[i] for i in range(len(conved1))]
        conved2 = [conved2[i] + convedTo2[i] for i in range(len(conved2))]

        cosine_value1 = [self.cosine_2(conved1[i], conved2[i]).mean(dim=1, keepdim=True) for i in range(len(conved1))]
        # print(cosine_value1[0].size())
        
        self.cosine_value1 = torch.cat(cosine_value1, dim=1).mean(dim=1)
        # print(cosine_value1.size())
        # conved2 = [F.relu(conv(x2)).squeeze(3) for conv in self.convs2]
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        pooled1 = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved1]
        #pooled_n = [batch size, n_filters]
        pooled2 = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved2]
        # S_pool = self.get_crosssim(pooled1, pooled2) #batchsize x len(conved2)
        paras_pool = self.get_parameters(pooled1, dim=1)
        attens_pool1 = [self.get_crossatten2(pooled1[i], pooled2[i], paras_pool[i]) for i in range(len(pooled1))]
        attens_pool2 = [self.get_crossatten2(pooled2[i], pooled1[i], paras_pool[i]) for i in range(len(pooled1))]

        convedTo1 = [torch.mm(attens_pool2[i], pooled2[i]) for i in range(len(pooled2))]
        convedTo2 = [torch.mm(attens_pool1[i], pooled1[i]) for i in range(len(pooled1))]

        # convedTo1 = [F.softmax(pooled2[i], dim=1)*pooled2[i] for i in range(len(pooled2))]
        # convedTo2 = [F.softmax(pooled1[i], dim=1)*pooled1[i] for i in range(len(pooled1))]

        pooled1 = [pooled1[i] + convedTo1[i] for i in range(len(pooled1))]
        pooled2 = [pooled2[i] + convedTo2[i] for i in range(len(pooled2))]


        out1 = torch.cat(pooled1, dim = 1)
        out2 = torch.cat(pooled2, dim = 1) 
        self.cosine_value2 = self.cosine_1(out1, out2)

        out1 = self.dropout(out1)
        out2 = self.dropout(out2)

 
        att1 = self.sigmoid(out1)


 
        att2 = self.sigmoid(out2)
        # att2 = F.softmax(out2, dim=1)
        # att2_1 = torch.exp(out2)
        # att2 = att2_1/ (1e-8 + (att2_1.sum(dim=1, keepdim=True)))

        out1 = out1*att2#*x1_att
        out2 = out2*att1#*x2_att
        self.cosine_value3 = self.cosine_1(out1, out2)
        # out = torch.cat([out1, out2], dim=1)
        # out = out1 + out2

        out_sent = self.fcSent(out1)
        out_PDN = self.fcPDN(out2)
        out_PDN_pose = self.sigmoid(out_PDN)
        # out = (self.fc(out))

        return out_sent, out_PDN, out_PDN_pose
class SiGMTL(nn.Module):
    def __init__(self, args, device):
        super(Sigmoid_MTL, self).__init__()

        self.embedding_dim = args.embedding_dim
        n_filters = 256
        filter_sizes = [3, 4, 5]
        output_dim = 768
        dropout = 0.5
        self.convs1 = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, self.embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        self.convs2 = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, self.embedding_dim)) 
                                    for fs in filter_sizes
                                    ])        
        self.fc = nn.Linear(output_dim*2, output_dim)
        self.gate1 = nn.Linear(len(filter_sizes) * n_filters , output_dim)
        self.gate2 = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()  

        self.fcPDN = nn.Linear(768, 5)

        self.fcSent = nn.Linear(768, 7)
        self.cosine_1 = nn.CosineSimilarity(dim=1)
        self.cosine_2 = nn.CosineSimilarity(dim=2)
 



    def forward(self, x1, x2):
        ## x1 batch size, sent len, emb dim
 

        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)

        # x = torch.cat([x1, x2], dim =1)
        # print(x.size(), x1.size())

        # x = x.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        conved1 = [F.relu(conv(x1)).squeeze(3) for conv in self.convs1]

        conved2 = [F.relu(conv(x2)).squeeze(3) for conv in self.convs2]


        convedTo1 = [self.sigmoid(conv)  for conv in conved2]
        convedTo2 = [self.sigmoid(conv)  for conv in conved1]

        conved1 = [conved1[i] + convedTo1[i] for i in range(len(conved1))]
        conved2 = [conved2[i] + convedTo2[i] for i in range(len(conved2))]

        cosine_value1 = [self.cosine_2(conved1[i], conved2[i]).mean(dim=1, keepdim=True) for i in range(len(conved1))]
        # print(cosine_value1[0].size())
        
        self.cosine_value1 = torch.cat(cosine_value1, dim=1).mean(dim=1)
        # print(cosine_value1.size())
        # conved2 = [F.relu(conv(x2)).squeeze(3) for conv in self.convs2]
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        pooled1 = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved1]
        #pooled_n = [batch size, n_filters]
        pooled2 = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved2]

        convedTo1 = [self.sigmoid(conv) for conv in pooled2]
        convedTo2 = [self.sigmoid(conv) for conv in pooled1]

        pooled1 = [pooled1[i] + convedTo1[i] for i in range(len(pooled1))]
        pooled2 = [pooled2[i] + convedTo2[i] for i in range(len(pooled2))]


        out1 = torch.cat(pooled1, dim = 1)
        out2 = torch.cat(pooled2, dim = 1) 
        self.cosine_value2 = self.cosine_1(out1, out2)

        out1 = self.dropout(out1)
        out2 = self.dropout(out2)

 
        att1 = self.sigmoid(out1)


 

        # gate2 = self.sigmoid(self.gate2(out2_sig))
        att2 = self.sigmoid(out2)
        # att2 = F.softmax(out2, dim=1)
        # att2_1 = torch.exp(out2)
        # att2 = att2_1/ (1e-8 + (att2_1.sum(dim=1, keepdim=True)))

        out1 = out1*att2#*x1_att
        out2 = out2*att1#*x2_att
        self.cosine_value3 = self.cosine_1(out1, out2)
        # out = torch.cat([out1, out2], dim=1)
        # out = out1 + out2

        out_sent = self.fcSent(out1)
        out_PDN = self.fcPDN(out2)
        out_PDN_pose = self.sigmoid(out_PDN)
        # out = (self.fc(out))

        return out_sent, out_PDN, out_PDN_pose

class SoGMTL(nn.Module):
    def __init__(self, args, device):
        super(CNN_MTL, self).__init__()

        self.embedding_dim = args.embedding_dim
        n_filters = 256
        # n_filters_PDN = 128
        filter_sizes = [3, 4, 5]
        filter_sizes_PDN = [3, 4, 5]
        output_dim = 768
        dropout = 0.5
        self.args = args
        self.convs1 = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, self.embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        self.convs2 = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, self.embedding_dim)) 
                                    for fs in filter_sizes_PDN
                                    ])       
        self.conv_inner = nn.Conv1d(in_channels = self.embedding_dim, 
            out_channels = self.embedding_dim, kernel_size = 3)
        # self.conv_inner = nn.Conv2d(in_channels = 1, out_channels = self.embedding_dim,
        #     kernel_size = (3, self.embedding_dim))

        self.fc = nn.Linear(output_dim*2, output_dim)
        self.gate1 = nn.Linear(len(filter_sizes) * n_filters , output_dim)
        self.gate2 = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.linear_att = nn.Linear(self.embedding_dim *2, self.embedding_dim, bias = False)
        self.linear_general = nn.Linear(self.embedding_dim, self.embedding_dim, bias = False)
        
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()  

        self.fcPDN = nn.Linear(768, 5)

        self.fcSent = nn.Linear(768, 7)
        self.cosine_1 = nn.CosineSimilarity(dim=1)
        self.cosine_2 = nn.CosineSimilarity(dim=2)

        if args.SeqLen_Max_PDN>args.SeqLen_Max_Sent:
            # self.trans = nn.Linear(args.SeqLen_Max_PDN, args.SeqLen_Max_Sent)
            self.W = nn.Parameter(torch.FloatTensor(args.SeqLen_Max_PDN, args.SeqLen_Max_Sent)).cuda()
        else:
            # self.trans = nn.Linear(args.SeqLen_Max_Sent, args.SeqLen_Max_PDN)

            self.W = nn.Parameter(torch.FloatTensor(args.SeqLen_Max_Sent, args.SeqLen_Max_PDN)).cuda()
        self.tanh = nn.Tanh()

        self.W1 = nn.Parameter(torch.FloatTensor(10, 1)).cuda()


        torch.nn.init.kaiming_normal_(self.W1)
 

    def forward(self, x1, x2):
 
        def att_sum(x):

            query = torch.mean(x, dim=1, keepdim=True)
            # query = self.linear_general(query)
            query_len = x.size()[1]
            att_scores = torch.bmm(query, x.transpose(1,2).contiguous()) ## batch_size x 1 x query_len
            # att_scores = att.view(-1, query_len)
            att_weights = F.softmax(att_scores, dim = 1) 
            output = torch.bmm(att_weights, x)  ## ## batch_size x 1 x embeding_dim
            # combined = torch.cat((output, query), dim=2)
            # output = self.linear_att(combined) 
            # output = self.tanh(output)           

            # x = torch.sum(x*a, dim=1, keepdim=True)
            return mix
        def inner_conv(x):
            x = x.permute(0, 2, 1) ## batch_size x embedding_dim x seq_len
            # x = x.unsqueeze(1) ## batch_size x 1 x seq_len x embedding_dim

            conv = self.conv_inner(x)  ## batch_size x self.embedding x seq_len - filter x 1
            # print("conv",conv.size())## batch_size x embedding_dim_out x seq_len-filter
            conv = F.relu(conv)
            pooled = F.max_pool1d(conv, conv.shape[2])## batch_size x embedding_dim_out x 1
            # print(pooled.size())
            pooled = pooled.permute(0, 2, 1)
            return pooled
        def mul(x):
            x = x.permute(0, 2, 1)
            out = x.matmul(self.W1)
            out = out.permute(0, 2, 1)
            return out

        # splitNum = int(np.ceil(x2.size()[1] / x1.size()[1]))
        # x2 = torch.split(x2, splitNum, dim=1) 
        # x2 = torch.cat([torch.mean(x, dim=1, keepdim=True) for x in x2], dim =1)
        # x2 = torch.cat([inn(x, dim=1, keepdim=True) for x in x2], dim =1)
        # print(x1.size(), x2.size())
        
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)

        # x = torch.cat([x1, x2], dim =1)
        # print(x2.size(), x1.size())

        # x = x.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        conved1 = [F.relu(conv(x1)).squeeze(3) for conv in self.convs1]
        # conved1.extend(conved1)

        conved2 = [F.relu(conv(x2)).squeeze(3) for conv in self.convs2]


        convedTo1 = [F.softmax(conv, dim=1)*conv  for conv in conved2]
        convedTo2 = [F.softmax(conv, dim=1)*conv  for conv in conved1]

        conved1 = [conved1[i] + convedTo1[i] for i in range(len(conved1))]
        conved2 = [conved2[i] + convedTo2[i] for i in range(len(conved2))]
        # conved1 = [torch.cat((conved1[i], convedTo1[i]), dim=2) for i in range(len(conved1))]
        # conved2 = [torch.cat((conved2[i], convedTo2[i]), dim=2) for i in range(len(conved2))]
        cosine_value1 = [self.cosine_2(conved1[i], conved2[i]).mean(dim=1, keepdim=True) for i in range(len(conved1))]
        # print(cosine_value1[0].size())
        
        self.cosine_value1 = torch.cat(cosine_value1, dim=1).mean(dim=1)
        # print(cosine_value1.size())
        # conved2 = [F.relu(conv(x2)).squeeze(3) for conv in self.convs2]
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        pooled1 = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved1]
        #pooled_n = [batch size, n_filters]
        pooled2 = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved2]

        convedTo1 = [F.softmax(conv, dim=1)*conv for conv in pooled2]
        convedTo2 = [F.softmax(conv, dim=1)*conv for conv in pooled1]

        pooled1 = [pooled1[i] + convedTo1[i] for i in range(len(pooled1))]
        pooled2 = [pooled2[i] + convedTo2[i] for i in range(len(pooled2))]


        out1 = torch.cat(pooled1, dim = 1)
        out2 = torch.cat(pooled2, dim = 1) 
        self.cosine_value2 = self.cosine_1(out1, out2)

        out1 = self.dropout(out1)
        out2 = self.dropout(out2)



        out1 = self.dropout(torch.cat(pooled1, dim = 1))
     
        att1 = self.sigmoid(out1)


 
        att2 = self.sigmoid(out2)
        # att2 = F.softmax(out2, dim=1)
        # att2_1 = torch.exp(out2)
        # att2 = att2_1/ (1e-8 + (att2_1.sum(dim=1, keepdim=True)))

        out1 = out1*att2#*x1_att
        out2 = out2*att1#*x2_att
        self.cosine_value3 = self.cosine_1(out1, out2)
        # out = torch.cat([out1, out2], dim=1)
        # out = out1 + out2

        out_sent = self.fcSent(out1)
        out_PDN = self.fcPDN(out2)
        out_PDN_pose = self.sigmoid(out_PDN)
        # out = (self.fc(out))

        return out_sent, out_PDN, out_PDN_pose

class CNN_base(nn.Module):
    def __init__(self, args, device, vocab_size, pre_embedding):
    # def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 # dropout, pad_idx):
        
        super().__init__()
        self.embedding_dim = args.embedding_dim

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.embedding.weight = nn.Parameter(torch.from_numpy(pre_embedding), requires_grad=True) 

        n_filters = 256
        filter_sizes = [3, 4, 5]
        output_dim = 7
        dropout = 0.5
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, self.embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, hidden):
         
        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
            
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        out = self.dropout(torch.cat(pooled, dim = 1))

        #cat = [batch size, n_filters * len(filter_sizes)]
        # out = self.fc(cat)
        return out, hidden
    def init_hidden(self, batch_size):
        self.n_layers=1
        batch_size=10
        self.hidden_dim=1
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),#.to(self.device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())#.to(self.device))
        return hidden
class LSTM_base(nn.Module):
    def __init__(self, args, device, vocab_size, pre_embedding):
        super(LSTM_base, self).__init__()
        self.output_size = args.output_size
        self.n_layers = args.n_layers
        self.hidden_dim = 768#args.hidden_dim
        self.device =  device
        self.seq_len = args.seq_length_given
        self.embedding = nn.Embedding(vocab_size, args.embedding_dim)


        self.embedding.weight = nn.Parameter(torch.from_numpy(pre_embedding), requires_grad=True) 
        self.lstm = nn.LSTM(args.embedding_dim, self.hidden_dim, self.n_layers , dropout=args.drop_prob, batch_first=True)

        # self.fc1 = nn.Linear(self.hidden_dim , 32)
        self.dropout = nn.Dropout(args.drop_prob)
        # self.fc2 = nn.Linear(32, 768)
        # self.fc3 = nn.Linear(64, self.output_size)
        # self.softmax = nn.Softmax()
        # self.sigmoid = nn.Sigmoid()  
    def forward(self, x, hidden): 
        x = x.long()
        embeds = self.embedding(x)
       
    
        batch_size = x.size()[0]

        lstm_out, hidden = self.lstm(embeds, hidden) 
        lstm_out = lstm_out.contiguous().view(batch_size, self.seq_len, self.hidden_dim)
        lstm_out = lstm_out[:,-1,:]
    
        # out = self.fc1(lstm_out)
        # out = F.relu(out)
        out = self.dropout(lstm_out)
 
        return out, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device))
        return hidden
