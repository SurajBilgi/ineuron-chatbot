from tensorflow import keras
n_enc_model = keras.models.load_model('models/my_enc_model.h5')
n_dec_model = keras.models.load_model('models/my_dec_model.h5')

import numpy as np


from keras_preprocessing.sequence import pad_sequences


import pandas as pd
import json

df=pd.read_json("messages.json")

user=[]
support=[]
len1=len(df['conversations'])
for k in range(0,len1):
  flag=0
  conv_len=len(df['conversations'][k]['MessageList'])
  for i in range(0,conv_len):
    Msg_From=df['conversations'][k]['MessageList'][i]['displayName'] 
    if Msg_From!=None:
      if flag==2 or flag==0 :
        flag=1
        user.append(str(df['conversations'][k]['MessageList'][i]['content']))
      else:
        user[-1]=user[-1]+","+ str(df['conversations'][k]['MessageList'][i]['content'])
    else:
      if flag==1 or flag==0:
        flag=2
        support.append(str(df['conversations'][k]['MessageList'][i]['content']))
      else:
        support[-1]=support[-1]+","+str(df['conversations'][k]['MessageList'][i]['content'])

  if len(user)>len(support):
    user.pop()
  elif len(user)<len(support):
    support.pop()
  else:
    pass

parsed_df=pd.DataFrame(data=[user,support],index=["user","support"]).T
parsed_df.to_csv("parsed_df.csv")
del(df)

parsed_df=parsed_df.drop(parsed_df.index[parsed_df['user'].str.contains('<') == True])
parsed_df=parsed_df.drop(parsed_df.index[parsed_df['support'].str.contains('<') == True])

user=parsed_df['user'].tolist()[:25000]
support=parsed_df['support'].tolist()[:25000]

word2count = {}

for line in user:
    for word in line.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
for line in support:
    for word in line.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

del(word, line)

thresh = 5
vocab = {}
word_num = 0
for word, count in word2count.items():
    if count >= thresh:
        vocab[word] = word_num
        word_num += 1
## delete
del(word2count, word, count, thresh)       
del(word_num)

for i in range(len(support)):
    support[i] = '<SOS> ' + support[i] + ' <EOS>'



tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
x = len(vocab)
for token in tokens:
    vocab[token] = x
    x += 1


vocab['cameron'] = vocab['<PAD>']
vocab['<PAD>'] = 0

## delete
del(token, tokens) 
del(x)

### inv answers dict ###
inv_vocab = {w:v for v, w in vocab.items()}



encoder_inp = []
for line in user:
    lst = []
    for word in line.split():
        if word not in vocab:
            lst.append(vocab['<OUT>'])
        else:
            lst.append(vocab[word])
        
    encoder_inp.append(lst)

decoder_inp = []
for line in support:
    lst = []
    for word in line.split():
        if word not in vocab:
            lst.append(vocab['<OUT>'])
        else:
            lst.append(vocab[word])        
    decoder_inp.append(lst)

### delete
del(support, user, line, lst, word)

### Model Building
from tensorflow.keras.preprocessing.sequence import pad_sequences
encoder_inp = pad_sequences(encoder_inp, 13, padding='post', truncating='post')
decoder_inp = pad_sequences(decoder_inp, 13, padding='post', truncating='post')

decoder_final_output = []
for i in decoder_inp:
    decoder_final_output.append(i[1:]) 

decoder_final_output = pad_sequences(decoder_final_output, 13, padding='post', truncating='post')
del(i)

from tensorflow.keras.utils import to_categorical
decoder_final_output = to_categorical(decoder_final_output, len(vocab))
#print(decoder_final_output.shape)


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input


enc_inp = Input(shape=(13, ))
dec_inp = Input(shape=(13, ))


VOCAB_SIZE = len(vocab)
embed = Embedding(VOCAB_SIZE+1, output_dim=50, 
                  input_length=13,
                  trainable=True                  
                  )


enc_embed = embed(enc_inp)
enc_lstm = LSTM(400, return_sequences=True, return_state=True)
enc_op, h, c = enc_lstm(enc_embed)
enc_states = [h, c]



dec_embed = embed(dec_inp)
dec_lstm = LSTM(400, return_sequences=True, return_state=True)
dec_op, _, _ = dec_lstm(dec_embed, initial_state=enc_states)

dense = Dense(VOCAB_SIZE, activation='softmax')

dense_op = dense(dec_op)

model = Model([enc_inp, dec_inp], dense_op)




# model.compile(loss='categorical_crossentropy',metrics=['acc'],optimizer='adam')

# model.fit([encoder_inp, decoder_inp],decoder_final_output,epochs=100)

# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input


# enc_model = Model([enc_inp], enc_states)
# enc_model.save("models/my_enc_model.h5")


# # decoder Model
# decoder_state_input_h = Input(shape=(400,))
# decoder_state_input_c = Input(shape=(400,))

# decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]


# decoder_outputs, state_h, state_c = dec_lstm(dec_embed , 
#                                     initial_state=decoder_states_inputs)


# decoder_states = [state_h, state_c]


# dec_model = Model([dec_inp]+ decoder_states_inputs,
#                                       [decoder_outputs]+ decoder_states)
# dec_model.save("models/my_dec_model.h5")

from tensorflow import keras
n_enc_model = keras.models.load_model('models/my_enc_model.h5')
n_dec_model = keras.models.load_model('models/my_dec_model.h5')


def chatbot_function(input_word):
    prepro1 = ""
    while prepro1 != 'q':
        prepro1  = input_word
        prepro = [prepro1]

        txt = []
        for x in prepro:
            lst = []
            for y in x.split():
                
                try:
                    lst.append(vocab[y])
                    
                except:
                    lst.append(vocab['<OUT>'])
            txt.append(lst)

        txt = pad_sequences(txt, 13, padding='post')
        stat = n_enc_model.predict( txt )

        empty_target_seq = np.zeros( ( 1 , 1) )
        ##   empty_target_seq = [0]
        empty_target_seq[0, 0] = vocab['<SOS>']
        stop_condition = False
        decoded_translation = ''

        while not stop_condition :

            dec_outputs , h, c= n_dec_model.predict([ empty_target_seq] + stat )
            decoder_concat_input = dense(dec_outputs)

            sampled_word_index = np.argmax( decoder_concat_input[0, -1, :] )

            sampled_word = inv_vocab[sampled_word_index] + ' '


            if sampled_word != '<EOS> ':
                decoded_translation += sampled_word  

            if sampled_word == '<EOS> ' or len(decoded_translation.split()) > 13:
                stop_condition = True 

            empty_target_seq = np.zeros( ( 1 , 1 ) )  
            empty_target_seq[ 0 , 0 ] = sampled_word_index

            stat = [h, c]  

        return decoded_translation
        # print("chatbot attention : ", decoded_translation )
        # print("==============================================")



# chatbot_function("Hi Sir")