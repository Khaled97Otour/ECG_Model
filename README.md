# ECG_Model
A CNN to predict the Abnormal ECG signals.
### I collect the data from local hospitals the data was preprocess , normalized and saperated into two files(one for each class) 
## Algorithm :
- building dataset after cleaning all the signals and establish a dataframe as well as define a colume of labels.
- labeling the input data and build the steps.
- I use a reshape function to build a signal 188 to 10.
- I take steps each one langth was 18.
### Building model:
Rnn model was built using TimeDistributed(10 steps) and 1d CNN model to extract the features.

Model: sequential
### Layer (type)                Output Shape              Param   
=================================================================
 time_distributed (TimeDistr  (None, 10, 256)          353024    
 ibuted)                                                         
                                                                 
 lstm (LSTM)                 (None, 64)                82176     
                                                                 
 dense (Dense)               (None, 1024)              66560     
                                                                 
 batch_normalization (BatchN  (None, 1024)             4096      
 ormalization)                                                   
                                                                 
 dense_1 (Dense)             (None, 512)               524800    
                                                                 
 batch_normalization_1 (Batc  (None, 512)              2048      
 hNormalization)                                                 
                                                                 
 dense_2 (Dense)             (None, 256)               131328    
                                                                 
 batch_normalization_2 (Batc  (None, 256)              1024      
 hNormalization)                                                 
                                                                 
 dense_3 (Dense)             (None, 128)               32896     
                                                                 
 batch_normalization_3 (Batc  (None, 128)              512       
 hNormalization)                                                 
                                                                 
 dense_4 (Dense)             (None, 64)                8256      
                                                                 
 batch_normalization_4 (Batc  (None, 64)               256       
 hNormalization)                                                 
                                                                 
 dense_5 (Dense)             (None, 32)                2080      
                                                                 
 batch_normalization_5 (Batc  (None, 32)               128       
 hNormalization)                                                 
                                                                 
 dropout (Dropout)           (None, 32)                0         
                                                                 
 dense_6 (Dense)             (None, 1)                 33        
                                                                 
=================================================================
Total params: 1,209,217
Trainable params: 1,205,185
Non-trainable params: 4,032
_________________________________________________________________

None

## Results:
52/52 [==============================] - 2s 31ms/step - loss: 0.0763 - accuracy: 0.9776 - val_loss: 0.0919 - val_accuracy: 0.9695

model.evaluate:
 loss: 0.1072 - accuracy: 0.9650
 
 
![download](https://user-images.githubusercontent.com/93203143/188266842-86736ced-9739-48f3-941a-c8c2cbabc9bb.png)
![download (1)](https://user-images.githubusercontent.com/93203143/188266844-e1b20014-8510-4009-92f6-c0d5a3edb12e.png)


