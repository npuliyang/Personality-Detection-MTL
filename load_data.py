#!/usr/bin/env python
# from Dataset_STS import dataset as STS
from emotion_isear import dataset as ISEAR
from emotion_tec import dataset as TEC
from Dataset_personality import dataset as PND
# from Dataset_essays import dataset as PND
import numpy as np
import random
from pytorch_pretrained_bert import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_word_dic():
	new_set = ISEAR + PND +TEC
	words_count = {}
	words_count['UNK'] = 1e6
	index = 1
	seq_lengths = [len(sent[0].strip().split(" ")) for sent in new_set]
	seq_lengths_max = max(seq_lengths)
	seq_lengths_avg = np.mean(seq_lengths)
	for sent_pair in new_set:
		sent = sent_pair[0].strip().split(" ")
		for w in sent:
			if w not in words_count:
				words_count[w] = 1
			else:
				words_count[w] += 1
	count_pairs = sorted(words_count.items(), key=lambda x: -x[1])
	words_all, _ = zip(*count_pairs)	
	dict_word = dict(zip(words_all, range(len(words_all))))
	dict_word_reverse = dict(zip(range(len(words_all)), words_all))
	return dict_word, dict_word_reverse, seq_lengths_max, seq_lengths_avg	
def load_data(args): 

	# with codecs.open(file_train, "r") as f:
	# 	data = f.read().strip().replace("\n"," ").split(' ') ## There must have space " " in this line , other wise it will have None item in the word onehot list
 		
	# with codecs.open(file_test, "r") as f:
	# 	data.extend(f.read().strip().replace("\n"," ").split(' ')) 

	dict_word, dict_word_reverse, seq_lengths_max, seq_lengths_avg = get_word_dic()
	
 
	vocab_size = len(dict_word)  
 
	print("==== vocab_size is ", (vocab_size))

	print("==== Max sentence Len is ",seq_lengths_max)
	print("==== Avg sentence Len is ",seq_lengths_avg)
	if args.SeqLen_Max_Sent>0:
		seq_lengths_max = args.SeqLen_Max_Sent
 

	if args.max_num == None:
		max_num =-1
	else:
		max_num = args.max_num
	

	def padding(sentence, seq_lengths_max):
		if len(sentence)<=seq_lengths_max:
			for k in range(seq_lengths_max - len(sentence)):
				sentence.insert(0, dict_word['UNK'])
			# sentence.extend([dict_word['UNK']]*(seq_lengths_max-len(sentence)))
		else:
			sentence = sentence[:seq_lengths_max]
			# assert True, "the seqence leng the is bigger than defined."
		return sentence
 
	def map_words(dict_word, words, BERT):
		
		if BERT:
			# stops = [".", "?","!"]
			# words_new = ['[CLS]']
			# print(len(words))
			# for word in words: 

			# 	if len(word)>0 and word[-1] in stops:
			# 		words_new.append(word[:-1])
			# 		words_new.append('[SEP]') 
		 
			# 	else:
			# 		words_new.append(word)
			token_text = tokenizer.tokenize(" ".join(words))
			# print(token_text)
			sent = tokenizer.convert_tokens_to_ids(token_text)
		else:
			sent = []
			for word in words:
			# if BERT:
			# 	# sent = tokenizer.encod
			# 	try:
			# 		indexed_token = tokenizer.convert_tokens_to_ids([word]) 
			# 		sent.extend(indexed_token)
			# 	except:
			# 		indexed_token = tokenizer.convert_tokens_to_ids(["[UNK]"]) 
			# 		sent.extend(indexed_token)
			# else:
				if word in dict_word:
					sent.append(dict_word[word])
				else:
					sent.append(dict_word["UNK"])

		return sent

	def dataset_process(dataset, dict_word, seq_lengths_max, shuffle=False, split=False):


		if shuffle:
			random.shuffle(dataset) 
		else:
			pass		
		train_sent = [] 
		train_label = [] 
		for sent_pair in dataset: 
			words = sent_pair[0].strip().split(" ")
			label = sent_pair[1:]
			
				 
			sent_words = map_words(dict_word, words, False)
			sent_words = padding(sent_words, seq_lengths_max) 
			 
			train_label.append(label)

			train_sent.append(sent_words) 

		if split:

			train_num = int(0.9*len(train_sent)) 
			train_sent_new = train_sent[:train_num]
			train_label_new = train_label[:train_num]
			test_sent = train_sent[train_num:]
			test_label = train_label[train_num:]
			return ((train_sent_new, train_label_new), (test_sent, test_label))
		else:
			return (train_sent, train_label)


	Sent_Train, Sent_Test = dataset_process(TEC, dict_word, args.SeqLen_Max_Sent, True, True)
	PND_Train, PND_Test = dataset_process(PND, dict_word, args.SeqLen_Max_PDN, True, True)

	 
	print ("the shape of the Sent set is ", np.array(Sent_Train[0]).shape,np.array(Sent_Test[1]).shape,np.array(Sent_Train[1]).shape)
	print ("the shape of the PND set is ", np.array(PND_Train[0]).shape,  np.array(PND_Test[1]).shape, np.array(PND_Train[1]).shape)
	 
	return Sent_Train, Sent_Test, PND_Train, PND_Test, dict_word, dict_word_reverse, vocab_size


# dict_word, dict_word_reverse, seq_lengths_max, seq_lengths_avg = get_word_dic()
# load_data("personality")