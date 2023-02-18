# Dissertation - Novel NER model for Product Domain
Repo for my dissertation that I did as part of my MSc Data Science. I developed a new Named Entity Recognition model for the product domain by manual annotation and creating training data, data pre-processing, and modeling using python and spacy for NLP. 

The product dataset is taken from ISWC-2020 (International Semantic Web Challenge) which is formatted from various online and retail website that sell product online. I annotated the dataset myself with the help of my friend to annotate BRAND and PRODUCT entities in the data. We were able to succesfully annotate approx. 2500 instances of product titles.

**Tech Stack:** Tensorflow, Keras, Neural Network, Conditional Random Fields (CRF), BiLSTM, Spacy, pandas, NumPy, sci-kit learn.

## EDA
The word cloud of the most frequent brand names and products in the dataset is as below. 
<p align='center'>
  <img src='https://user-images.githubusercontent.com/60603790/217555204-a7a4786b-212f-4769-b3d5-dd7132bda3ee.png' width=400 height=250 />
  <img src='https://user-images.githubusercontent.com/60603790/217557216-68c0c3bf-0cae-4908-9362-b2c9ab4dc577.png' width=400 height=250 />
 </p>

## Process
#### Convert to BIO tags
The product titles and their corresponding entity tags are converted into SpaCy format before casting them through to BIO tagging. The output of SpaCy training format is as below.

[['sterling angel charm',
  {'entities': [(0, 8, 'BRAND'), (15, 20, 'PRODUCT')]}],
  
 ['hp pavilion 23xi 5840 cm 23 ips monitor',
  {'entities': [(0, 2, 'BRAND'), (32, 39, 'PRODUCT')]}]]
 
**Syntax:** [['product_title', {'entities': [(start_character_of_entity, end_character_of_entity, 'entity')]}]]

Then data from previous format are converted BIO tagging format with the 'offsets_to_biluo_tags' function from the SpaCy library.

#### Word Embeddings
The 'Word2Vec' function from the Gensim library if used to generate word embeddings for the corpus. Now, word embedding for each word is extracted and also an 'UNK' token is appended for the words that are not in the corpus. A word embedding is generated for UNk as well. Final input feature data contained the sentence id, word embedding for each word, its pos tag, corresponding target tag (product, brand, or other). 

#### Neural Network
A one hidden layered Neural Network with 'relu' activation function was developed. The Input, Dense,Embedding, Activation, Flatten layers from the TensorFLow library was selected for this purpose. The classification report of the model is as below.
<p align='center'>
  <img src='https://user-images.githubusercontent.com/60603790/217574423-76cda7f7-4ae9-4983-b212-9a6ed1da7deb.png' width=400 height=250 />
 </p>

#### BiLSTM
The LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional layers from keras library were used to build a BiLSTM model with one embedding layer, bidirectional layer and time distributed layer with softmax activation function. The model needs all the length of each instance to be same hence 'ENDPAD' was used to equalize all the lengths to the maximum length instance in the corpus. The classification report of the BiLSTM model is as below:
<p align='center'>
  <img src='https://user-images.githubusercontent.com/60603790/217567061-3f6b208b-35b9-4ebf-bcb0-5765b0d4f040.png' width=400 height=250 />
 </p>

Please feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/niteesh-chanabasanavar/)
