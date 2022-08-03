# ECG_Model
A CNN to predict the Abnormal ECG signals.
### I collect the data from local hospitals 
## Algorithm :
### First Signal pre processing :
in this function I use for loops to help me filtering the signals and delete the high frequnce noice taking in mind the range of the signals and the length 
- side_note: this process could be used in any code to help clean any ECG dataset. 
### Second Do dataset:
- building dataset after cleaning all the signals and establish a dataframe as well as define a colume of labels.
- labeling the input data and build the steps.
- I use a reshape function to build a signal 188 to 10.
- I take steps each one langth was 18.
### third building model:
Rnn model was built using TimeDistributed(10 steps) and 1d CNN model to extract the features.

Model: sequential_1
### Layer (type)                Output Shape              Param   
=================================================================

 time_distributed_1 (TimeDi  (None, 10, 256)          363072    
 stributed)                                                      
                                                                 
 lstm_1 (LSTM)              (None, 64)                82176     
                                                                 
 dense_1 (Dense)           (None, 1024)              66560     
                                                                 
 dense_2 (Dense)           (None, 512)               524800    
                                                                 
 dense_3 (Dense)           (None, 256)               131328    
                                                                 
 dense_4 (Dense)           (None, 128)               32896     
                                                                 
 dense_5 (Dense)           (None, 64)                8256      
                                                                 
 dense_6 (Dense)           (None, 32)                2080      
                                                                 
 batch_normalization_1 (Ba  (None, 32)               128       
 tchNormalization)                                               
                                                                 
 dropout_1 (Dropout)       (None, 32)                0         
                                                                 
 dense_7 (Dense)           (None, 2)                 66        
                                                                 
=================================================================

Total params: 1,211,362

Trainable params: 1,211,298

Non-trainable params: 64
_________________________________________________________________

None

## Results:
loss: 0.3073 - accuracy: 0.8916 - val_loss: 0.2991 - val_accuracy: 0.8965

![download](https://user-images.githubusercontent.com/93203143/182677259-e82673be-97fb-4f48-83ee-1783a3cc6266.png)
![download (1)](https://user-images.githubusercontent.com/93203143/182677264-0b077de0-4dea-4b63-b7ce-199ef03690aa.png)
