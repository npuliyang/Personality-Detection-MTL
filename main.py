# encoding = utf-8
import torch
import torch.nn as nn
import numpy as np
from GraphMTL import  Sent_NN, PDN_NN, BERT_base, CNN_base, LSTM_base
from GraphMTL import MTL_Input, SoGMTL, SiLGMTL, CAGMTL, SiGMTL
from load_data import load_data#, batch_helper
from w2v import load_word2vec
import argparse
import json
import random 
import time
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import f1_score

parser = argparse.ArgumentParser()

 
parser.add_argument('--embedding_dim', type=int, default=300, help='embedding size')
parser.add_argument('--SeqLen_Max_Sent', type=int, default=95, help='the length of sentences') 
parser.add_argument('--SeqLen_Max_PDN', type=int, default=1500, help='the length of sentences') 
parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=30, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for training')
parser.add_argument('--class_dim', type=int, default=1, help='The dimension of the class')
parser.add_argument('--output_size', type=int, default=1, help='The dimension of the output')
parser.add_argument('--bidirectional', type=bool, default=False, help='Using bidirectional rnn')
parser.add_argument('--n_layers', type=int, default=1, help='The number of the lstm layers')
parser.add_argument('--max_num', type=int, default=None, help='The max number of the vocabulary size ') 
parser.add_argument('--print_every', type=int, default=1, help='Print out the results every number steps')
parser.add_argument('--clip', type=float, default=5.0, help='The max grad values') 
parser.add_argument('--load_exist', type=bool, default=False, help='Loading the existing trained parameters')
parser.add_argument('--state_save', type=str, default='state_dict.pt', help='Loading the existing trained parameters') 
parser.add_argument('--drop_prob', type=float, default=0.5, help='The dropout rate in the model during the training')
 
parser.add_argument('--hidden_dim', type=int, default=512, help='The hidden dimension of LSTM')
parser.add_argument('--learning_rate_decay_start', type=int, default=0, 
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
parser.add_argument('--learning_rate_decay_every', type=int, default=5, 
                    help='how many iterations thereafter to drop LR?(in epoch)')
parser.add_argument('--learning_rate_decay_rate', type=float, default=0.99, 
                    help='how many iterations thereafter to drop LR?(in epoch)')


args = parser.parse_args()
params = vars(args)
print(json.dumps(params, indent = 2))



def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def train_one_step_mix(model_shared, model_Sent, model_PDN, train_loaderSent, train_loaderPerson, args, device, optimizer, current_lr, criterion_xen, criterion_bce, epochs=1):
    train_data1, train_label1 = train_loaderSent
    train_data2, train_label2 = train_loaderPerson
    nr_train1 = len(train_data1)
    nr_train2 = len(train_data2)
    nr_train = nr_train1 if nr_train1< nr_train2 else nr_train2
    nr_train_batches = int(np.ceil(nr_train/float(args.batch_size)))
    train_loss = []


    for epoch in range(epochs):
        torch.cuda.empty_cache()
        if epoch > args.learning_rate_decay_start and args.learning_rate_decay_start >= 0:
            frac = (epoch - args.learning_rate_decay_start) // args.learning_rate_decay_every
            decay_factor = args.learning_rate_decay_rate  ** frac
            current_lr = current_lr * decay_factor
   
        set_lr(optimizer, current_lr) 
        # if model_type=="lstm":
        h = model_shared.init_hidden(args.batch_size) 
        for iteration, batch_idx in enumerate(np.random.permutation(range(nr_train_batches))):
            start_idx = batch_idx * args.batch_size
            end_idx = min((batch_idx + 1) * args.batch_size, nr_train)
            labels1 = torch.from_numpy(train_label1[start_idx:end_idx])
            inputs1 = torch.from_numpy(train_data1[start_idx:end_idx])

            labels2 = torch.from_numpy(train_label2[start_idx:end_idx])
            inputs2 = torch.from_numpy(train_data2[start_idx:end_idx])

            if len(inputs1)==args.batch_size: 
                h = tuple([e.data for e in h])
                inputs1, labels1 = inputs1.to(device), labels1.to(device)
                inputs2, labels2 = inputs2.to(device), labels2.to(device)
                optimizer.zero_grad() 

                shared_vec1, h = model_shared(inputs1, h)
                output1 = model_Sent(shared_vec1)

                shared_vec2, h = model_shared(inputs2, h)
                output2, pose2 = model_PDN(shared_vec2)

                class_dim1 = output1.size()[1]
                class_dim2 = output2.size()[1]

                # if loss_type!="bce":
                    # label_ = labels.type(torch.int64)
                ## crossentropy batch_size x sentiment_class
                label_1 = labels1.type(torch.long)
                label_1 = label_1.to(device)
                lossSent = criterion_xen(output1, label_1.squeeze())

                ### bce batch_size x personality (5)
                label_2 = labels2.type(torch.FloatTensor)
                label_2 = label_2.to(device)
                lossPDN = criterion_bce(output2.squeeze(), label_2)
                loss = lossSent + lossPDN
                # print(output.size(), inputs.size(), labels.size())

                
                train_loss.append(loss.item())
                loss.backward()
                nn.utils.clip_grad_norm_(model_Sent.parameters(), args.clip) 
                nn.utils.clip_grad_norm_(model_PDN.parameters(), args.clip) 
                nn.utils.clip_grad_norm_(model_shared.parameters(), args.clip) 
                optimizer.step()
    print("Train Loss: {:.6f}".format(np.mean(train_loss)))
def train_one_step_mtl_maml_self(model_input, model_shared, model_Sent, model_PDN, train_loaderSent, train_loaderPerson, args, device, optimizer, current_lr, criterion_xen, criterion_bce, epochs=1):
    train_data1, train_label1 = train_loaderSent
    train_data2, train_label2 = train_loaderPerson
    nr_train1 = len(train_data1)
    nr_train2 = len(train_data2)
    nr_train = nr_train1 if nr_train1 > nr_train2 else nr_train2
    nr_train_batches = int(np.ceil(nr_train/float(args.batch_size))) - 5
    train_loss = []


    for epoch in range(epochs):
        torch.cuda.empty_cache()
        if epoch > args.learning_rate_decay_start and args.learning_rate_decay_start >= 0:
            frac = (epoch - args.learning_rate_decay_start) // args.learning_rate_decay_every
            decay_factor = args.learning_rate_decay_rate  ** frac
            current_lr = current_lr * decay_factor
   
        set_lr(optimizer, current_lr) 
        # if model_type=="lstm":
        # h = model_shared.init_hidden(args.batch_size) 
 
        for iteration, batch_idx in enumerate(np.random.permutation(range(nr_train_batches))):
            start_idx = batch_idx * args.batch_size
            end_idx = min((batch_idx + 1) * args.batch_size, nr_train)
            labels1 = torch.from_numpy(train_label1[start_idx:end_idx])
            inputs1 = torch.from_numpy(train_data1[start_idx:end_idx])
            inputs1, labels1 = inputs1.to(device), labels1.to(device)
            input_vec1 = model_input(inputs1)
            # labels2 = torch.from_numpy(train_label2[start_idx:end_idx])
            # inputs2 = torch.from_numpy(train_data2[start_idx:end_idx])                

            parameters = list(model_shared.parameters())
            parameters.extend(model_Sent.parameters())
            parameters.extend(model_PDN.parameters())
            # parameters.extend(model_input.parameters())
            initial_values = []
            final_values = []
            losses = []
            scalar_losses = []
            cosine1 = []
            cosine2 = []
            cosine3 = []

            for k in range(1):
                optimizer.zero_grad() 
                # start_idx2 = start_idx + k * args.batch_size
                # end_idx2 = end_idx + k * args.batch_size 
                if end_idx>=nr_train2:
                    batch_idx_i = int((end_idx - int(end_idx/nr_train2)*nr_train2)/args.batch_size)
                    start_idx2 = (batch_idx_i + k) * args.batch_size
                    end_idx2 = (batch_idx_i + 1 + k) * args.batch_size
                    if end_idx2 > nr_train2:
                        start_idx2 = 0 + k*args.batch_size
                        end_idx2 = (k+1) * args.batch_size
                    # print(start_idx2, end_idx2)

                else:
                    start_idx2 = start_idx + k * args.batch_size
                    end_idx2 = end_idx + k * args.batch_size 
                    if end_idx2 > nr_train2:
                        start_idx2 = 0 +  k*args.batch_size
                        end_idx2 = (k+1) * args.batch_size
                # labels1 = torch.from_numpy(train_label1[start_idx2:end_idx2])
                # inputs1 = torch.from_numpy(train_data1[start_idx2:end_idx2])
                # inputs1, labels1 = inputs1.to(device), labels1.to(device)
                labels2 = torch.from_numpy(train_label2[start_idx2:end_idx2])
                inputs2 = torch.from_numpy(train_data2[start_idx2:end_idx2])    
                inputs2, labels2 = inputs2.to(device), labels2.to(device)
                assert inputs2.size(0)==args.batch_size
                

                
                input_vec2 = model_input(inputs2)
                output1, output2, pose2 = model_shared(input_vec1, input_vec2)
                # output1 = model_Sent(shared_vec)
                # output2, pose2 = model_PDN(shared_vec)
                cosine_value1 = model_shared.cosine_value1.mean()
                cosine_value2 = model_shared.cosine_value2.mean()
                cosine_value3 = model_shared.cosine_value3.mean()
                cosine1.append(cosine_value1.item())
                cosine2.append(cosine_value2.item())
                cosine3.append(cosine_value3.item())

                class_dim1 = output1.size()[1]
                class_dim2 = output2.size()[1]
                label_1 = labels1.type(torch.long)
                label_1 = label_1.to(device)
                lossSent = criterion_xen(output1, label_1.squeeze())
                ### bce batch_size x personality (5)
                label_2 = labels2.type(torch.FloatTensor)
                label_2 = label_2.to(device)
                lossPDN = criterion_bce(output2.squeeze(), label_2)
                # if epoch % 2==0:
                loss = lossSent + lossPDN# + cosine_value1 - cosine_value2
                if lossSent > 3*lossPDN:
                    alpha = 3
                else:
                    alpha = 1
                # else:
                #     loss = lossSent# + 0.45*lossPDN

                losses.append(loss)

                initial_values.append([p.clone().detach() for p in parameters])
                updated = []
                # print(parameters)
                grads = torch.autograd.grad(loss, parameters, create_graph=True, retain_graph=True, allow_unused=True)
                # print(grads)
                for grad, param in zip(grads, parameters):
                    if grad is not None:
                        x = param - current_lr * grad
                        updated.append(x)
                        param.data.copy_(x)
                    else:
                        updated.append(param)

                final_values.append(updated)

            
            gradient = [torch.zeros_like(p) for p in parameters]
            for loss, initial, final in list(zip(losses, initial_values, final_values))[::-1]:
                for p, x in zip(parameters, initial):
                    p.data.copy_(x)

                grad1 = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
                grad2 = torch.autograd.grad(final, parameters, grad_outputs=gradient, retain_graph=True, allow_unused=True)

                gradient = []
                for ki in range(len(grad1)):
                    v1 = grad1[ki]
                    v2 = grad2[ki]
                    if v1 is None:
                        gradient.append(v2)
                    elif v2 is None:
                        gradient.append(v1)
                    else:
                        gradient.append(v1 + 3*v2)

                # gradient = [v1 + v2 for v1, v2 in zip(grad1, grad2)]
            for p, g in zip(parameters, gradient):
                if p.grad is None:
                    p.grad = g
                else:
                    p.grad.add_(g)




            # label_1 = labels1.type(torch.long)
            # label_1 = label_1.to(device)
            # lossSent = criterion_xen(output1, label_1.squeeze())
            # loss = 50*lossSent + lossPDN
                # print(output.size(), inputs.size(), labels.size())

                
            train_loss.append(loss.item())
            # loss.backward()
            nn.utils.clip_grad_norm_(model_Sent.parameters(), args.clip) 
            nn.utils.clip_grad_norm_(model_PDN.parameters(), args.clip) 
            nn.utils.clip_grad_norm_(model_shared.parameters(), args.clip) 
            optimizer.step()
    print("Train Loss: {:.6f}".format(np.mean(train_loss)),
        "cosine1: {:.5f}".format(np.mean(cosine1)),
        "cosine2: {:.5f}".format(np.mean(cosine2)),
        "cosine3: {:.5f}".format(np.mean(cosine3)),
        )

def train_one_step_mtl_maml(model_input, model_shared, model_Sent, model_PDN, train_loaderSent, train_loaderPerson, args, device, optimizer, current_lr, criterion_xen, criterion_bce, epochs=1):
    train_data1, train_label1 = train_loaderSent
    train_data2, train_label2 = train_loaderPerson
    nr_train1 = len(train_data1)
    nr_train2 = len(train_data2)
    nr_train = nr_train1 if nr_train1 > nr_train2 else nr_train2
    nr_train_batches = int(np.ceil(nr_train/float(args.batch_size)))  
    train_loss = []


    for epoch in range(epochs):
        torch.cuda.empty_cache()
        if epoch > args.learning_rate_decay_start and args.learning_rate_decay_start >= 0:
            frac = (epoch - args.learning_rate_decay_start) // args.learning_rate_decay_every
            decay_factor = args.learning_rate_decay_rate  ** frac
            current_lr = current_lr * decay_factor
   
        set_lr(optimizer, current_lr) 
        # if model_type=="lstm":
        # h = model_shared.init_hidden(args.batch_size) 
 
        for iteration, batch_idx in enumerate(np.random.permutation(range(nr_train_batches))):
            start_idx = batch_idx * args.batch_size
            end_idx = min((batch_idx + 1) * args.batch_size, nr_train)
            labels1 = torch.from_numpy(train_label1[start_idx:end_idx])
            inputs1 = torch.from_numpy(train_data1[start_idx:end_idx])
            inputs1, labels1 = inputs1.to(device), labels1.to(device)

            # labels2 = torch.from_numpy(train_label2[start_idx:end_idx])
            # inputs2 = torch.from_numpy(train_data2[start_idx:end_idx])                

            parameters = list(model_shared.parameters())
            parameters.extend(model_Sent.parameters())
            parameters.extend(model_PDN.parameters())
            initial_values = []
            final_values = []
            losses = []
            scalar_losses = []
            cosine1 = []
            cosine2 = []
            for k in range(1):
                optimizer.zero_grad() 
                if end_idx>=nr_train2:
                    batch_idx_i = int((end_idx - int(end_idx/nr_train2)*nr_train2)/args.batch_size)
                    start_idx2 = (batch_idx_i) * args.batch_size
                    end_idx2 = (batch_idx_i + 1) * args.batch_size
                    if end_idx_i > nr_train2:
                        start_idx2 = 0
                        end_idx2 = args.batch_size
                    print(start_idx2, end_idx2)

                else:
                    start_idx2 = start_idx + k * args.batch_size
                    end_idx2 = end_idx + k * args.batch_size 


                # start_idx2 = start_idx + k * args.batch_size
                # end_idx2 = end_idx + k * args.batch_size 
                # labels1 = torch.from_numpy(train_label1[start_idx2:end_idx2])
                # inputs1 = torch.from_numpy(train_data1[start_idx2:end_idx2])
                # inputs1, labels1 = inputs1.to(device), labels1.to(device)
                labels2 = torch.from_numpy(train_label2[start_idx2:end_idx2])
                inputs2 = torch.from_numpy(train_data2[start_idx2:end_idx2])    
                inputs2, labels2 = inputs2.to(device), labels2.to(device)
                

                input_vec1 = model_input(inputs1)
                input_vec2 = model_input(inputs2)
                output1, output2, pose2 = model_shared(input_vec1, input_vec2)
                # output1 = model_Sent(shared_vec)
                # output2, pose2 = model_PDN(shared_vec)
                cosine_value1 = model_shared.cosine_value1.mean()
                cosine_value2 = model_shared.cosine_value2.mean()
                cosine1.append(cosine_value1.item())
                cosine2.append(cosine_value2.item())

                class_dim1 = output1.size()[1]
                class_dim2 = output2.size()[1]
                label_1 = labels1.type(torch.long)
                label_1 = label_1.to(device)
                lossSent = criterion_xen(output1, label_1.squeeze())
                ### bce batch_size x personality (5)
                label_2 = labels2.type(torch.FloatTensor)
                label_2 = label_2.to(device)
                lossPDN = criterion_bce(output2.squeeze(), label_2)
                # if epoch % 2==0:
                loss = 0.8*lossSent + 0.2*lossPDN# + cosine_value1 - cosine_value2
                # else:
                #     loss = lossSent# + 0.45*lossPDN

                losses.append(loss)

                initial_values.append([p.clone().detach() for p in parameters])
                updated = []
                # print(parameters)
                grads = torch.autograd.grad(loss, parameters, create_graph=True, retain_graph=True, allow_unused=True)
                # print(grads)
                for grad, param in zip(grads, parameters):
                    if grad is not None:
                        x = param - current_lr * grad
                        updated.append(x)
                        param.data.copy_(x)
                    else:
                        updated.append(param)

                final_values.append(updated)

            
            gradient = [torch.zeros_like(p) for p in parameters]
            for loss, initial, final in list(zip(losses, initial_values, final_values))[::-1]:
                for p, x in zip(parameters, initial):
                    p.data.copy_(x)

                grad1 = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
                grad2 = torch.autograd.grad(final, parameters, grad_outputs=gradient, retain_graph=True, allow_unused=True)

                gradient = []
                for ki in range(len(grad1)):
                    v1 = grad1[ki]
                    v2 = grad2[ki]
                    if v1 is None:
                        gradient.append(v2)
                    elif v2 is None:
                        gradient.append(v1)
                    else:
                        gradient.append(v1+v2)
                # gradient = [v1 + v2 for v1, v2 in zip(grad1, grad2)]
            for p, g in zip(parameters, gradient):
                if p.grad is None:
                    p.grad = g
                else:
                    p.grad.add_(g)




            # label_1 = labels1.type(torch.long)
            # label_1 = label_1.to(device)
            # lossSent = criterion_xen(output1, label_1.squeeze())
            # loss = 50*lossSent + lossPDN
                # print(output.size(), inputs.size(), labels.size())

                
            train_loss.append(loss.item())
            # loss.backward()
            nn.utils.clip_grad_norm_(model_Sent.parameters(), args.clip) 
            nn.utils.clip_grad_norm_(model_PDN.parameters(), args.clip) 
            nn.utils.clip_grad_norm_(model_shared.parameters(), args.clip) 
            optimizer.step()
    print("Train Loss: {:.6f}".format(np.mean(train_loss)),
        "cosine1:{:.5f}".format(np.mean(cosine1)),
        "cosine2:{:.5f}".format(np.mean(cosine2)),
        )
def train_one_step_mtl(model_input, model_shared, model_Sent, model_PDN, train_loaderSent, train_loaderPerson, args, device, optimizer, current_lr, criterion_xen, criterion_bce, epochs=1):
    train_data1, train_label1 = train_loaderSent
    train_data2, train_label2 = train_loaderPerson
    nr_train1 = len(train_data1)
    nr_train2 = len(train_data2)
    nr_train = nr_train1 if nr_train1 > nr_train2 else nr_train2
    nr_train_batches = int(np.floor(nr_train/float(args.batch_size))) 
    train_loss = []


    for epoch in range(epochs):
        torch.cuda.empty_cache()
        if epoch > args.learning_rate_decay_start and args.learning_rate_decay_start >= 0:
            frac = (epoch - args.learning_rate_decay_start) // args.learning_rate_decay_every
            decay_factor = args.learning_rate_decay_rate  ** frac
            current_lr = current_lr * decay_factor
   
        set_lr(optimizer, current_lr) 
        # if model_type=="lstm":
        # h = model_shared.init_hidden(args.batch_size) 
 
        for iteration, batch_idx in enumerate(np.random.permutation(range(nr_train_batches))):
            start_idx = batch_idx * args.batch_size
            end_idx = min((batch_idx + 1) * args.batch_size, nr_train)
            labels1 = torch.from_numpy(train_label1[start_idx:end_idx])
            inputs1 = torch.from_numpy(train_data1[start_idx:end_idx])
            if end_idx>=nr_train2:
                batch_idx_i = int((end_idx - int(end_idx/nr_train2)*nr_train2)/args.batch_size)
                start_idx_i = (batch_idx_i) * args.batch_size
                end_idx_i = (batch_idx_i + 1) * args.batch_size
                if end_idx_i > nr_train2:
                    start_idx_i = 0
                    end_idx_i = args.batch_size
                # print(start_idx_i, end_idx_i, batch_idx_i, end_idx)

            else:
                start_idx_i = start_idx
                end_idx_i = end_idx
            
            labels2 = torch.from_numpy(train_label2[start_idx_i:end_idx_i])
            inputs2 = torch.from_numpy(train_data2[start_idx_i:end_idx_i])
            # if              

       
            scalar_losses = []
            cosine1 = []
            cosine2 = []
            # for k in range(5):
            optimizer.zero_grad() 
                # start_idx2 = start_idx + k * args.batch_size
                # end_idx2 = end_idx + k * args.batch_size 
                # labels1 = torch.from_numpy(train_label1[start_idx2:end_idx2])
                # inputs1 = torch.from_numpy(train_data1[start_idx2:end_idx2])
            inputs1, labels1 = inputs1.to(device), labels1.to(device)
                # labels2 = torch.from_numpy(train_label2[start_idx2:end_idx2])
                # inputs2 = torch.from_numpy(train_data2[start_idx2:end_idx2])    
            inputs2, labels2 = inputs2.to(device), labels2.to(device)
                

            input_vec1 = model_input(inputs1)
            input_vec2 = model_input(inputs2)
            output1, output2, pose2 = model_shared(input_vec1, input_vec2)
                # output1 = model_Sent(shared_vec)
                # output2, pose2 = model_PDN(shared_vec)
            cosine_value1 = model_shared.cosine_value1.mean()
            cosine_value2 = model_shared.cosine_value2.mean()
            cosine1.append(cosine_value1.item())
            cosine2.append(cosine_value2.item())

            class_dim1 = output1.size()[1]
            class_dim2 = output2.size()[1]
            label_1 = labels1.type(torch.long)
            label_1 = label_1.to(device)
            lossSent = criterion_xen(output1, label_1.squeeze())
                ### bce batch_size x personality (5)
            label_2 = labels2.type(torch.FloatTensor)
            label_2 = label_2.to(device)
            lossPDN = criterion_bce(output2.squeeze(), label_2)
            # if epoch % 2==0:
            loss = lossSent + lossPDN# + cosine_value1 - cosine_value2
                # else:
                #     loss = lossSent# + 0.45*lossPDN

            # losses.append(loss)
 
                
            train_loss.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model_Sent.parameters(), args.clip) 
            nn.utils.clip_grad_norm_(model_PDN.parameters(), args.clip) 
            nn.utils.clip_grad_norm_(model_shared.parameters(), args.clip) 
            optimizer.step()
    print("Train Loss: {:.6f}".format(np.mean(train_loss)),
        "cosine1:{:.5f}".format(np.mean(cosine1)),
        "cosine2:{:.5f}".format(np.mean(cosine2)),
        )



def train_one_step(model_shared, model, train_loader, args, device, optimizer, current_lr, criterion, loss_type="bce", epochs=1, model_type = "lstm"):
    train_data, train_label = train_loader
    nr_train = len(train_data)
    nr_train_batches = int(np.ceil(nr_train/float(args.batch_size)))
    train_loss = []


    for epoch in range(epochs):
        torch.cuda.empty_cache()
        if epoch > args.learning_rate_decay_start and args.learning_rate_decay_start >= 0:
            frac = (epoch - args.learning_rate_decay_start) // args.learning_rate_decay_every
            decay_factor = args.learning_rate_decay_rate  ** frac
            current_lr = current_lr * decay_factor
   
        set_lr(optimizer, current_lr) 
        # if model_type=="lstm":
        h = model_shared.init_hidden(args.batch_size) 
        for iteration, batch_idx in enumerate(np.random.permutation(range(nr_train_batches))):
            start_idx = batch_idx * args.batch_size
            end_idx = min((batch_idx + 1) * args.batch_size, nr_train)
            labels = torch.from_numpy(train_label[start_idx:end_idx])
            inputs = torch.from_numpy(train_data[start_idx:end_idx])

            if len(inputs)==args.batch_size: 
                h = tuple([e.data for e in h])
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad() 

                shared_vec, h = model_shared(inputs, h)
                if loss_type=="xen":
                    output = model(shared_vec)
                else: ## bce
                    output, pose = model(shared_vec)
                # print(output)
                class_dim = output.size()[1]
                if loss_type!="bce":
                    # label_ = labels.type(torch.int64)
                    label_ = labels.type(torch.long)
                    label_ = label_.to(device)

                    # label_onehot = torch.FloatTensor(args.batch_size, class_dim)
                    # label_onehot.zero_()
                    # label_onehot.scatter_(1, label_, 1)
                    # print(label_.size())
                    loss = criterion(output, label_.squeeze())

                else:

                    label_ = labels.type(torch.FloatTensor)
                    label_ = label_.to(device)
                    loss = criterion(output.squeeze(), label_)

                # print(output.size(), inputs.size(), labels.size())

                
                train_loss.append(loss.item())
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip) 
                nn.utils.clip_grad_norm_(model_shared.parameters(), args.clip) 
                optimizer.step()
    print("Train Loss: {:.6f}".format(np.mean(train_loss)))
def test_one_step(model_shared, model, test_loader, args, device, test_loss_min, criterion, class_dim=2):

    totoal = 0
    correct = 0
        
    test_h = model_shared.init_hidden(args.batch_size)
    test_losses = []
    model_shared.eval()
    model.eval()
    test_data, test_label = test_loader
    nr_test = len(test_data)
    nr_test_batches = int(np.ceil(nr_test)/float(args.batch_size))  
    # accs = []
    for step_test in range(nr_test_batches):
        start_idx = step_test*args.batch_size
        end_idx = (step_test+1)*args.batch_size
        inp = torch.from_numpy(test_data[start_idx:end_idx])
        lab = torch.from_numpy(test_label[start_idx:end_idx])

        if len(inp)==args.batch_size:
            test_h = tuple([each.data for each in test_h])
            inp, lab = inp.to(device), lab.to(device) 
            shared_vec, test_h = model_shared(inp, test_h)
            out = model(shared_vec)

                    
            if class_dim==2:
                tru_lab = lab.type(torch.FloatTensor)

                pre_lab = (out>0.5)
                test_loss = criterion(out.squeeze(), tru_lab)

            else:
                tru_lab = lab.type(torch.long).squeeze()
                # print(out.size(), tru_lab.size())
                pre_lab = out.argmax(dim=1).long()
                test_loss = criterion(out, tru_lab)
            tru_lab_byte = tru_lab.type(torch.long).to(device)

            totoal += out.size(0)
            correct += (pre_lab==tru_lab).sum().item()


            # acc_sk = accuracy_score(tru_lab.item(), pre_lab.item())
            # accs.append(acc_sk)
            test_losses.append(test_loss.item())
                    
            model_shared.train()
            model.train()
    print("Test Loss: {:.6f}".format(np.mean(test_losses)),
            "Test Acc: {:.6f}".format(correct/totoal))
    # print(np.mean(accs))
    if np.mean(test_losses) <= test_loss_min:
        torch.save({'model_shared': model_shared.state_dict(),
            'model_Sent': model.state_dict(),
            }, args.state_save)
        # torch.save(model_shared.state_dict(), args.state_save)
        # torch.save(model.state_dict(), args.state_save)
        print('Test loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(test_loss_min,np.mean(test_losses)))
        test_loss_min = np.mean(test_losses)
 
    torch.cuda.empty_cache()
    return test_loss_min
def test_one_step_mtl(model_input, model_shared, model_Sent, model_PDN, test_loaderSent, 
    test_loaderPDN, test_loss_min, criterionSent, criterionPDN, dict_word_reverse):
 
    # test_h = model_shared.init_hidden(args.batch_size)
    test_losses = []
    lossesPDN = []
    lossesSent = []
    model_input.eval()
    model_shared.eval()
    model_Sent.eval()
    model_PDN.eval()
 
    total = 0
    correctSent = 0 
    correctPDN = 0  
    correctsPDN = 0  
    correct_1 = 0
    correct_2 = 0
    correct_3 = 0
    correct_4 = 0
    correct_5 = 0

    TP_all = 0
    TN_all = 0
    FP_all = 0
    FN_all = 0

    test_data_sent, test_label_sent = test_loaderSent
    test_data_PDN, test_label_PDN = test_loaderPDN

    test_data_len = len(test_data_sent) if len(test_data_PDN) > len(test_data_sent) else len(test_data_PDN)

    nr_test_batchesSent = int(np.floor(test_data_len/float(args.batch_size))) 

    for step_test in range(nr_test_batchesSent):
        start_idx = step_test*args.batch_size
        end_idx = (step_test+1)*args.batch_size
        inp_sent = torch.from_numpy(test_data_sent[start_idx:end_idx])
        lab_sent = torch.from_numpy(test_label_sent[start_idx:end_idx])
        inp_sent, lab_sent = inp_sent.to(device), lab_sent.to(device) 

        inp_PDN = torch.from_numpy(test_data_PDN[start_idx:end_idx])
        lab_PDN = torch.from_numpy(test_label_PDN[start_idx:end_idx])
        inp_PDN, lab_PDN = inp_PDN.to(device), lab_PDN.to(device) 

        inp_vecSent = model_input(inp_sent)
        inp_vecPDN = model_input(inp_PDN)

        outSent, outPDN, pose = model_shared(inp_vecSent, inp_vecPDN)

        # outSent = model_Sent(shared_vec)
        # outPDN, pose = model_PDN(shared_vec)
        class_dim = outSent.size()[1]
        # print(class_dim)
        if class_dim==2:
            tru_labSent = lab_sent.type(torch.FloatTensor)
            pre_labSent = (outSent>0.5)
            test_lossSent = criterion(outSent.squeeze(), tru_labSent)

        else:

            tru_labSent = lab_sent.type(torch.long).squeeze()
            # print(outSent.size(), tru_labSent.size())

            pre_labSent = outSent.argmax(dim=1).long()

            test_lossSent = criterionSent(outSent, tru_labSent)

        tru_labSent_byte = tru_labSent.type(torch.long).to(device)
        total += outSent.size(0)
        correctSent += (pre_labSent==tru_labSent).sum().item() 




        tru_labPDN = lab_PDN.type(torch.FloatTensor).to(device) 
        tru_labPDN_byte = tru_labPDN.type(torch.long).to(device)
        pre_labPDN = (pose>0.5).long()
        # totoal += pose.size()[0]
        correctPDN += (pre_labPDN==tru_labPDN_byte).sum(dim=0).sum().item()
        case_num = 0
        correctsPDN = (pre_labPDN==tru_labPDN_byte).sum(dim=0) 
        # print(correctsPDN.size())
        correct_1 += correctsPDN[0].item()
        correct_2 += correctsPDN[1].item()
        correct_3 += correctsPDN[2].item()
        correct_4 += correctsPDN[3].item()
        correct_5 += correctsPDN[4].item()
        ### case study ####
        # for i in range(20):

        #     case_PDN = inp_PDN[i]
        #     case_sent = inp_sent[i]
        #     case_PDN_Label = lab_PDN[i].cpu().numpy() 
        #     case_sent_label = lab_sent[i][0].item()
        #     case_pre_PDN = pre_labPDN[i].cpu().numpy()

        #     case_PDN_test = sum([case_PDN_Label[j]== case_pre_PDN[j] for j in range(5)])
        #     case_pre_Sent = pre_labSent[i].item()
        #     if case_sent_label == case_pre_Sent and case_PDN_test == 5 and case_num<=5:
        #         case_num+=1

        #         case_study(case_sent, case_PDN, dict_word_reverse)
        #         print("true emotion label: ",case_sent_label)
        #         print("predict emotion label: ",outSent[i])
        #         print("true personality label: ",case_PDN_Label)
        #         print("predict personality label: ",pose[i]) 


        TP_all += ((pre_labPDN==tru_labPDN_byte).long()*pre_labPDN).sum(dim=0).sum().item()
        TN_all += ((pre_labPDN==tru_labPDN_byte).long()*(1-pre_labPDN)).sum(dim=0).sum().item()
        FP_all += ((pre_labPDN!=tru_labPDN_byte).long()*pre_labPDN).sum(dim=0).sum().item()
        FN_all += ((pre_labPDN!=tru_labPDN_byte).long()*(1-pre_labPDN)).sum(dim=0).sum().item()

        test_lossPDN = criterionPDN(outPDN.squeeze(), tru_labPDN)
        test_losses.append(test_lossPDN.item() +  test_lossSent.item())
        lossesPDN.append(test_lossPDN.item())
        lossesSent.append(test_lossSent.item())
    Acc_1 = correct_1/(1.0*total)
    Acc_2 = correct_2/(1.0*total)
    Acc_3 = correct_3/(1.0*total)
    Acc_4 = correct_4/(1.0*total)
    Acc_5 = correct_5/(1.0*total)
    AccSent = correctSent / (1.0*total)
    AccPDN = (correctPDN/(1.0*total))/5
    Precision = TP_all/(1.0*(TP_all+FP_all))
    Recall = TP_all/(1.0*(FP_all+FN_all))
    F1 = 2.0*Precision*Recall/(Precision+Recall)      
    torch.cuda.empty_cache()
    if np.mean(test_losses) <= test_loss_min:
        torch.save({'model_shared': model_shared.state_dict(),
                'model_Sent': model_Sent.state_dict(),
                'model_PDN': model_PDN.state_dict(),
                'model_input': model_input.state_dict()
                }, args.state_save)
            # torch.save(model_shared.state_dict(), args.state_save)
            # torch.save(model.state_dict(), args.state_save)
        print('Test loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(test_loss_min,np.mean(test_losses)))
        test_loss_min = np.mean(test_losses)
    print("LossPDN: {:.5f}".format(np.mean(lossesPDN)),
         "LossSent: {:.5f}".format(np.mean(lossesSent)))
    print("Test Loss: {:.5f}".format(np.mean(test_losses)),
        "AccSent: {:.5f}\n".format(AccSent),
        "Test AccPDN: {:.5f}".format(AccPDN),
        "Precision: {:.5f}".format(Precision),
        "Recall: {:.5f}".format(Recall),
        "F1: {:.5f}".format(F1),
        )
    print("EXT: {:.6f}".format(Acc_1),#ext, cneu, carg, ccon, copn
            "NEU: {:.6f}".format(Acc_2),
            "ARG: {:.6f}".format(Acc_3),
            "CON: {:.6f}".format(Acc_4),
            "OPN: {:.6f}".format(Acc_5),
            )
    model_input.train()
    model_shared.train()
    model_Sent.train()
    model_PDN.train()

    return test_loss_min 

def case_study(sent, pdnsent, dict_word_reverse):
    sentstr = [dict_word_reverse[word.item()] for word in sent if word.item()!=0]
    sentstr = " ".join(sentstr)
    pdnsentstr = [dict_word_reverse[word.item()] for word in pdnsent if word.item()!=0]
    pdnsentstr = " ".join(pdnsentstr)
    print(sentstr)
    print(pdnsentstr)

    # test_h = model_shared.init_hidden(args.batch_size)
  

def test_one_step_multilabel(model_shared, model, test_loader, args, device, test_loss_min, criterion):


    totoal = 0
    correct = np.array([0,0,0,0,0])
    correct_1 = 0
    correct_2 = 0
    correct_3 = 0
    correct_4 = 0
    correct_5 = 0
        
    test_h = model_shared.init_hidden(args.batch_size)
    test_losses = []
    model_shared.eval()
    model.eval()
    test_data, test_label = test_loader
    nr_test = len(test_data)
    nr_test_batches = int(np.ceil(nr_test)/float(args.batch_size))  
        

    TP_all = 0
    TN_all = 0
    FP_all = 0
    FN_all = 0
    corrects = 0
    for step_test in range(nr_test_batches):
        start_idx = step_test*args.batch_size
        end_idx = (step_test+1)*args.batch_size
        inp = torch.from_numpy(test_data[start_idx:end_idx])
        lab = torch.from_numpy(test_label[start_idx:end_idx])

        if len(inp)==args.batch_size:
            test_h = tuple([each.data for each in test_h])
            inp, lab = inp.to(device), lab.to(device) 
            shared_vec, test_h = model_shared(inp, test_h)
            out, pose = model(shared_vec)

                    
            tru_lab = lab.type(torch.FloatTensor).to(device) 
            tru_lab_byte = tru_lab.type(torch.long).to(device)
            pre_lab = (pose>0.5).long()
            totoal += pose.size()[0]
            corrects += (pre_lab==tru_lab_byte).sum(dim=0).sum().item()
            correct_1 += corrects[0].item()
            correct_2 += corrects[1].item()
            correct_3 += corrects[2].item()
            correct_4 += corrects[3].item()
            correct_5 += corrects[4].item()
            TP_all += ((pre_lab==tru_lab_byte).long()*pre_lab).sum(dim=0).sum().item()
            TN_all += ((pre_lab==tru_lab_byte).long()*(1-pre_lab)).sum(dim=0).sum().item()
            FP_all += ((pre_lab!=tru_lab_byte).long()*pre_lab).sum(dim=0).sum().item()
            FN_all += ((pre_lab!=tru_lab_byte).long()*(1-pre_lab)).sum(dim=0).sum().item()

            # print(totoal, corrects.size(),correct_1, corrects[0].item())

     


            test_loss = criterion(out.squeeze(), tru_lab)
            test_losses.append(test_loss.item())
                    
            model_shared.train()
            model.train()
    Acc = (corrects/(1.0*totoal))/5
    Acc_1 = correct_1/totoal
    Acc_2 = correct_2/totoal
    Acc_3 = correct_3/totoal
    Acc_4 = correct_4/totoal
    Acc_5 = correct_5/totoal
    Precision = TP_all/(1.0*(TP_all+FP_all))
    Recall = TP_all/(1.0*(FP_all+FN_all))
    F1 = 2*Precision*Recall/(Precision+Recall)

    print("Test Loss: {:.6f}".format(np.mean(test_losses)),
            "Test Acc: {:.6f}".format(Acc),
            "Test Pre: {:.6f}".format(Precision),
            "Test Rec: {:.6f}".format(Recall),
            "Test F1: {:.6f}".format(F1),
            # "Test Acc5: {:.6f}".format(Acc_5),
            )
    print("EXT: {:.6f}".format(Acc_1),#ext, cneu, carg, ccon, copn
            "NEU: {:.6f}".format(Acc_2),
            "ARG: {:.6f}".format(Acc_3),
            "CON: {:.6f}".format(Acc_4),
            "OPN: {:.6f}".format(Acc_5),
            )
    if np.mean(test_losses) <= test_loss_min:
        torch.save({'model_shared': model_shared.state_dict(),
            'model_PDN': model.state_dict(),
            }, args.state_save)
        # torch.save(model_shared.state_dict(), args.state_save)
        # torch.save(model.state_dict(), args.state_save)
        print('Test loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(test_loss_min,np.mean(test_losses)))
        test_loss_min = np.mean(test_losses)
 
    torch.cuda.empty_cache()
    return test_loss_min, (Acc, Precision, Recall, F1)


def trainer(model_input, model_shared, model_Sent, model_PDN, 
            Sent_Trloader, Sent_Teloader, PDN_Trloader, PDN_Teloader, args, device, dict_word_reverse):
    criterion_xen = nn.CrossEntropyLoss().to(device)
    # criterion_bce = nn.BCELoss().to(device)
    criterion_bce = nn.MultiLabelSoftMarginLoss().to(device)
    if args.load_exist: 
        checkpoint = torch.load(args.state_save)
        # para = torch.load(load_state_dict)

        print("Loading existing parameters")
        # embed = para["pretrain_sentiment.embedding.weight"]
        model_shared.load_state_dict(checkpoint['model_shared']) 
        if "model_Sent" in checkpoint:
            model_Sent.load_state_dict(checkpoint['model_Sent'])
        if 'model_PDN' in checkpoint:
            model_PDN.load_state_dict(checkpoint['model_PDN'])
    # model=nn.DataParallel(model,device_ids=[3,7]) # multi-GPU
    model_input.train()
    model_input.to(device)
    model_shared.train() 
    model_shared.to(device)
    model_Sent.train()
    model_Sent.to(device)
    model_PDN.train()
    model_PDN.to(device)
    # print(model)

    parameters = list(model_shared.parameters())
    parameters.extend(model_Sent.parameters())
    parameters.extend(model_PDN.parameters())
    # print(parameters)
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    
    counter = 0  
    valid_loss_min = np.Inf
    current_lr = args.learning_rate
    test_loss_minPDN = 1e3
    test_loss_minSent = 1e3
    test_loss_min = 1e3
    final = None

    for epoch in range(args.epochs):
        print(epoch)
        train_one_step_mtl_maml_self(model_input, model_shared, model_Sent, model_PDN, Sent_Trloader,
         PDN_Trloader, args, device, optimizer, current_lr, criterion_xen, criterion_bce, epochs=1)
        tic = time.time()
        test_loss_min = test_one_step_mtl(model_input, model_shared, model_Sent, model_PDN, 
            Sent_Teloader, PDN_Teloader, test_loss_min, criterion_xen, criterion_bce, dict_word_reverse)
        toc = time.time()
        print("The time cost", toc-tic)
        # train_one_step(model_shared, model_Sent, Sent_Trloader, args, device, optimizer, current_lr, criterion_xen, "xen")
        # test_loss_minSent = test_one_step(model_shared, model_Sent, Sent_Teloader, args, device, test_loss_minSent, criterion_xen, class_dim=7)

        # train_one_step(model_shared, model_PDN, PDN_Trloader, args, device, optimizer, current_lr, criterion_bce, "bce")
        # test_loss_minPDN, Accs = test_one_step_multilabel(model_shared, model_PDN, PDN_Teloader, args, device, test_loss_minPDN, criterion_bce)
        
        # if test_loss_min<test_loss_flag:
        #     final = Accs
        #     test_loss_flag = test_loss_min
    # return final, test_loss_flag

    
 
if __name__=="__main__":
    print("==== Loading Data ...")
    # train_set, test_set, dict_word, dict_word_reverse, seq_length_words, vocab_size = load_data(args)
    Sent_Train, Sent_Test, PND_Train, PND_Test, dict_word, dict_word_reverse, vocab_size = load_data(args)
    embedding_weights = load_word2vec('glove', dict_word, args.embedding_dim)

    # fw_res = open("res.txt","w")

    device = torch.device("cuda") 
    model_input = MTL_Input(args, device, vocab_size, embedding_weights)
    model_shared = SoGMTL(args, device)
    # model_shared = CNN_base(args, device, vocab_size, embedding_weights)
    model_Sent = Sent_NN(args, output_dim=6)
    model_PDN = PDN_NN(args, output_dim=5)
    model_parameters = []
    model_parameters_input = filter(lambda p: p.requires_grad, model_input.parameters())
    model_parameters_shared = filter(lambda p: p.requires_grad, model_shared.parameters())
    model_parameters_Sent = filter(lambda p: p.requires_grad, model_Sent.parameters())
    model_parameters_PDN = filter(lambda p: p.requires_grad, model_PDN.parameters())
    model_parameters.extend(model_parameters_input)
    model_parameters.extend(model_parameters_shared)
    model_parameters.extend(model_parameters_Sent)
    model_parameters.extend(model_parameters_input)
    model_parameters.extend(model_parameters_PDN)
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("The model parameters is", params)

    Sent_Trloader =  (np.array(Sent_Train[0]), np.array(Sent_Train[1]))
    Sent_Teloader =  (np.array(Sent_Test[0]), np.array(Sent_Test[1]))
    PND_Trloader =  (np.array(PND_Train[0]), np.array(PND_Train[1]))
    PND_Teloader =  (np.array(PND_Test[0]), np.array(PND_Test[1]))

    # for k in range(5):

    trainer(model_input, model_shared, model_Sent, model_PDN, 
            Sent_Trloader, Sent_Teloader, PND_Trloader, PND_Teloader, args, device, dict_word_reverse)
        # fw_res.write(str(test_loss)+" "+str(Accs)+"\n")
    