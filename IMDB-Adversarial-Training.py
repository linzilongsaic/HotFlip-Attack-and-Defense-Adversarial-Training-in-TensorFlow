from __future__ import absolute_import, division, print_function, unicode_literals
# from keras import backend as K
import tensorflow as tf
import keras
import numpy as np
# import tensorflow.contrib.eager as tfe
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import time
from keras.initializers import Constant
from helpers import loadGloveModel, evaluate_acc, hotflip_attack, get_pred, attack, attack_dataset

print(tf.__version__)


def batch_attack(m, input, targ, decode_review, loss_function):
    # print(input.shape)
    adv_output = np.zeros((input.shape[0], input.shape[1]))
    for idx, (data, label) in enumerate(zip(input, targ)):
        if idx % 1000 == 0:
            print("# query: {}".format(idx))
        o, sentence = attack(m, data.numpy(), label.numpy(), decode_review, loss_function, verbose=False)
        sentence = np.array(sentence[0])
        adv_output[idx] = sentence
    return adv_output


def evaluate_accuracy(labels, preds):
    total = len(labels)
    counter = 0
    for l, p in zip(labels, preds):
        dummy_l = l.numpy()
        dummy_p = tf.sigmoid(p).numpy()[0]
        if dummy_p >= 0.5:
            dummy_p = 1
        else:
            dummy_p = 0
        # print(dummy_l,dummy_p)
        if dummy_l == dummy_p:
            counter += 1
    return counter / total


def combined_loss_function(real, pred, pred_adv, a):
    pred = tf.reshape(pred, [-1])
    pred_adv = tf.reshape(pred_adv, [-1])
    loss_ = tf.keras.losses.BinaryCrossentropy(from_logits=True)(real, pred)
    loss_adv = tf.keras.losses.BinaryCrossentropy(from_logits=True)(real, pred_adv)
    final = tf.add(loss_, loss_adv)
    return final


if __name__ == '__main__':
    multitaskMode = True
    # 1.0 get the data
    # tfe.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)
    imdb = tf.keras.datasets.imdb
    num_features = 450000
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_features)
    # train_data = train_data[:2000]
    # train_labels = train_labels[:2000]
    print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
    maxlen = 80
    x_train = sequence.pad_sequences(train_data, maxlen=maxlen)
    x_test = sequence.pad_sequences(test_data, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    # 1.1 get the word indices
    word_index = imdb.get_word_index()
    # The first indices are reserved
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


    def decode_review(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])


    # 1.2 get the glove embedding
    Embedding_Dims = 50
    GLOVE_PATH = 'embeds/glove.6B.{}d.txt'.format(Embedding_Dims)
    glove = loadGloveModel(GLOVE_PATH)
    start_point = len(word_index)
    glove_words = list(glove.keys())
    for g_word in glove_words:
        if g_word not in word_index:
            word_index[g_word] = start_point
            start_point += 1
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    print("# word in reverse_word_index: {}".format(len(reverse_word_index)))

    glove_embedding = np.zeros(shape=(num_features, Embedding_Dims))
    for value in reverse_word_index:
        key = reverse_word_index[value]
        if key in glove:
            glove_embedding[value, :] = glove[key]
        else:
            glove_embedding[value, :] = np.random.uniform(size=(Embedding_Dims,))

    # 2.0 setup dataset and model parameters
    BUFFER_SIZE = len(x_train)
    BATCH_SIZE = 32
    N_BATCH = BUFFER_SIZE // BATCH_SIZE
    embedding_dims = Embedding_Dims
    lstm_units = 128


    # dataset = tf.data.Dataset.from_tensor_slices((x_train, train_labels)).shuffle(BUFFER_SIZE)
    # dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    # 2.1 define the model
    class Classifier(tf.keras.Model):
        def __init__(self, vocab_size, embedding_dim, lstm_units):
            super(Classifier, self).__init__()

            self.lstm_units = lstm_units
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                       embeddings_initializer=Constant(glove_embedding))
            self.lstm = tf.keras.layers.LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2)
            self.dense = tf.keras.layers.Dense(32, activation=tf.nn.relu)
            self.pred = tf.keras.layers.Dense(1, activation=None)

        def call(self, x, is_training):
            x = self.embedding(x)
            #         num_samples = tf.shape(x)[0]
            #         hidden = tf.zeros((BATCH_SIZE, self.lstm_units))
            #         print(self.lstm.get_initial_state(x))
            #         print(x)
            o = self.lstm(x)
            #         print(output.shape)
            #         o = tf.layers.dropout(o, rate=0.2, training=is_training)
            #         o = self.dense(o)
            o = self.pred(o)
            #         print(o)
            #         o = tf.nn.softmax(o)
            #         print(result)
            return o

    class Classifier2(tf.keras.Model):
        def __init__(self, vocab_size, embedding_dim, lstm_units):
            super(Classifier2, self).__init__()

            self.lstm_units = lstm_units
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                       embeddings_initializer=Constant(glove_embedding))
            self.gru = tf.keras.layers.GRU(lstm_units, dropout=0.2, recurrent_dropout=0.2)
            self.dense = tf.keras.layers.Dense(32, activation=tf.nn.relu)
            self.pred = tf.keras.layers.Dense(1, activation=None)

        def call(self, x, is_training):
            x = self.embedding(x)
            #         num_samples = tf.shape(x)[0]
            #         hidden = tf.zeros((BATCH_SIZE, self.lstm_units))
            #         print(self.lstm.get_initial_state(x))
            #         print(x)
            o = self.gru(x)
            #         print(output.shape)
            #         o = tf.layers.dropout(o, rate=0.2, training=is_training)
            #         o = self.dense(o)
            o = self.pred(o)
            #         print(o)
            #         o = tf.nn.softmax(o)
            #         print(result)
            return o


    # model = Classifier(num_features, embedding_dims, lstm_units)

    # 2.2 define loss
    optimizer = tf.optimizers.Adam()

    def loss_function(real, pred):
        # Reshape: [batch_size,class_size] -> [barch_size,]
        pred = tf.reshape(pred, [-1])
        loss_ = tf.keras.losses.BinaryCrossentropy(from_logits=True)(real, pred)
        # print("Single Loss: {}".format(loss_))
        return loss_


    # 2.3 train the model
    '''
    EPOCHS = 12

    for epoch in range(EPOCHS):
        start = time.time()
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(dataset):
            loss = 0
            with tf.GradientTape() as tape:
                output = model(inp,True)
    #             print(output)
        
                loss = loss_function(targ, output)
                    
            variables = model.trainable_variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
            
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                            batch,
                                                            loss))
    # 2.4 if there is already a model, load it                                                    
    checkpoint_prefix = "saved/orgM/"
    root = tf.train.Checkpoint(optimizer=optimizer,
                            model=model,
                            optimizer_step=tf.train.get_or_create_global_step())
    root.save(checkpoint_prefix)
    root.restore(tf.train.latest_checkpoint("saved/orgM/"))

    # 2.5 evaluate the model acc
    evaluate_acc(model,x_test,test_labels,BATCH_SIZE)
    '''

    # 3.3 naive adversarial training
    '''
    dataset_naive = tf.data.Dataset.from_tensor_slices((x_train2, train_labels2)).shuffle(BUFFER_SIZE)
    dataset_naive = dataset_naive.batch(BATCH_SIZE, drop_remainder=True)
    naive_adv_model = Classifier(num_features, embedding_dims, lstm_units)
    EPOCHS = 12
    for epoch in range(EPOCHS):
        start = time.time()
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(dataset_naive):
            loss = 0
            with tf.GradientTape() as tape:
                output = naive_adv_model(inp,True)
    #             print(output)
                loss = loss_function(targ, output)
                    
            variables = naive_adv_model.trainable_variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                            batch,
                                                            loss))
    print("the adversarial trained (naive) model's behavior on adversarial testset")
    evaluate_acc(naive_adv_model,x_test_adv,test_adv_labels2)
    '''

    # 3.4 non-naive adversarial training (may put in a seperate file)
    BUFFER_SIZE = len(x_train)
    BATCH_SIZE = 1024
    N_BATCH = BUFFER_SIZE // BATCH_SIZE
    print("N_BATCH: {}".format(N_BATCH))
    embedding_dims = Embedding_Dims
    lstm_units = 128

    dataset_comb = tf.data.Dataset.from_tensor_slices((x_train, train_labels)).shuffle(BUFFER_SIZE)
    dataset_comb = dataset_comb.batch(BATCH_SIZE, drop_remainder=True)
    comb_adv_model = Classifier(num_features, embedding_dims, lstm_units)
    if multitaskMode:
        comb_adv_model2 = Classifier2(num_features, embedding_dims, lstm_units)
    # optimizer = tf.train.AdamOptimizer()

    EPOCHS = 3  # 7
    print("Begin training! Epochs: %d, Num Batches: %d" % (EPOCHS, N_BATCH))

    for epoch in range(EPOCHS):
        start = time.time()
        total_loss = 0
        for batch, (inp, targ) in enumerate(dataset_comb):
            print("Epoch: ", epoch, "Batch:", batch)
            loss = 0
            if epoch >= 3:
                adv_output = batch_attack(comb_adv_model, inp, targ, decode_review, loss_function)
            with tf.GradientTape(persistent=True) as tape:
                output = comb_adv_model(inp, True)
                if multitaskMode:
                    output2 = comb_adv_model2(inp, True)

                if epoch >= 3:
                    r = comb_adv_model(adv_output, True)
                    loss = combined_loss_function(targ, output, r, 0.5)
                else:
                    print("Batch Accuracy:", evaluate_accuracy(targ, output))
                    loss = loss_function(targ, output)
                    if multitaskMode:
                        loss2 = loss_function(targ, output2)
                print("Batch Loss: {}".format(loss))
            print("------------->>")

            variables = comb_adv_model.trainable_variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
            if multitaskMode:
                variables2 = comb_adv_model2.trainable_variables
                gradients2 = tape.gradient(loss2, variables2)
                optimizer.apply_gradients(zip(gradients2, variables2))
                gradients += gradients2

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             loss))
        checkpoint_prefix = "saved/advM/"
        root = tf.train.Checkpoint(optimizer=optimizer,
                                   model=comb_adv_model,
                                   optimizer_step=tf.optimizers.SGD())
        root.save(checkpoint_prefix)
    checkpoint_prefix = "saved/advM/"
    root = tf.train.Checkpoint(optimizer=optimizer,
                               model=comb_adv_model,
                               optimizer_step=tf.optimizers.SGD())
    root.save(checkpoint_prefix)
    root.restore(tf.train.latest_checkpoint("saved/advM/"))

    if multitaskMode:
        checkpoint_prefix2 = "saved/advM2/"
        root2 = tf.train.Checkpoint(optimizer=optimizer,
                                   model=comb_adv_model2,
                                   optimizer_step=tf.optimizers.SGD())
        root2.save(checkpoint_prefix2)
        root2.restore(tf.train.latest_checkpoint("saved/advM2/"))
