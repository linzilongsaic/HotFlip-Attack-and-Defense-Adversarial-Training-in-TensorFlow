from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
# import tensorflow.contrib.eager as tfe
import keras
from keras.preprocessing import sequence
from keras.initializers import Constant
from helpers import loadGloveModel, evaluate_acc, hotflip_attack, get_pred, attack, attack_dataset,hotflip_attack_multitask
print(tf.__version__)


def batch_attack(m,input,targ):
    adv_output = np.zeros((input.shape[0],input.shape[1]))
    for idx,(data,label) in enumerate(zip(input,targ)):
        o,sentence = attack(m,data.numpy(),label.numpy(),verbose=False)
        sentence = np.array(sentence[0])
        adv_output[idx] = sentence
    return adv_output


def evaluate_accuracy(labels,preds):
    total = len(labels)
    counter = 0
    for l,p in zip(labels,preds):
        dummy_l = l.numpy()
        dummy_p = tf.sigmoid(p).numpy()[0]
        if dummy_p >=0.5:
            dummy_p = 1
        else:
            dummy_p = 0
        # print(dummy_l,dummy_p)
        if dummy_l==dummy_p:
            counter += 1
    return counter/total


def combined_loss_function(real,pred,pred_adv,a):
    pred = tf.reshape(pred, [-1])
    pred_adv = tf.reshape(pred_adv, [-1])
    loss_ = tf.keras.losses.BinaryCrossentropy(from_logits=True)(real, pred)
    loss_adv = tf.keras.losses.BinaryCrossentropy(from_logits=True)(real, pred_adv)
    print("Combined Loss: basic_{}, adv_{}".format(loss_,loss_adv))
    final = tf.add(loss_,loss_adv)
    return final


if __name__ == '__main__':
    multitaskMode = True

    # 1.0 get the data
    # tfe.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)
    imdb = tf.keras.datasets.imdb
    num_features = 450000
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_features)
    print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
    maxlen = 80
    x_train = sequence.pad_sequences(train_data, maxlen=maxlen)
    x_test = sequence.pad_sequences(test_data, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    # 1.1 get the word indices
    word_index = imdb.get_word_index()
    # The first indices are reserved
    word_index = {k:(v+3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


    def decode_review(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])

    # 1.2 get the glove embedding (big change)
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
    N_BATCH = BUFFER_SIZE//BATCH_SIZE
    embedding_dims = Embedding_Dims
    lstm_units = 128

    dataset = tf.data.Dataset.from_tensor_slices((x_train, train_labels)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    # 2.1 define the model
    class Classifier(tf.keras.Model):
        def __init__(self,vocab_size,embedding_dim,lstm_units):
            super(Classifier, self).__init__()
            
            self.lstm_units = lstm_units
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,embeddings_initializer=Constant(glove_embedding))
            self.lstm = tf.keras.layers.LSTM(lstm_units,dropout=0.2, recurrent_dropout=0.2)
            self.dense = tf.keras.layers.Dense(32,activation=tf.nn.relu)
            self.pred = tf.keras.layers.Dense(1,activation=None)
            

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
    # model = Classifier(num_features, embedding_dims, lstm_units)

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

    # 2.2 define loss
    optimizer = tf.optimizers.Adam()

    def loss_function(real, pred):
        # Reshape: [batch_size,class_size] -> [barch_size,]
        pred = tf.reshape(pred, [-1])
        loss_ = tf.keras.losses.BinaryCrossentropy(from_logits=True)(real, pred)
        # print("Single Loss: {}".format(loss_))
        return loss_
    '''
    # 2.4 if there is already a model, load it                                                    
    root = tf.train.Checkpoint(optimizer=optimizer,
                            model=model,
                            optimizer_step=tf.train.get_or_create_global_step())
    root.restore(tf.train.latest_checkpoint("saved/orgM/"))

    # 2.5 evaluate the model acc
    evaluate_acc(model,x_test,test_labels)
    '''
    # 3.1 load previous model
    BUFFER_SIZE = len(x_train)
    BATCH_SIZE = 64
    N_BATCH = BUFFER_SIZE//BATCH_SIZE
    embedding_dims = Embedding_Dims
    lstm_units = 128

    #dataset_comb = tf.data.Dataset.from_tensor_slices((x_test, test_labels)).shuffle(BUFFER_SIZE)
    #dataset_comb = dataset_comb.batch(BATCH_SIZE, drop_remainder=True)

    optimizer = tf.optimizers.Adam()

    comb_adv_model = Classifier(num_features, embedding_dims, lstm_units)
    root = tf.train.Checkpoint(optimizer=optimizer,
                            model=comb_adv_model,
                            optimizer_step=tf.optimizers.SGD())
    root.restore(tf.train.latest_checkpoint("saved/advM/"))

    if multitaskMode:
        comb_adv_model2 = Classifier2(num_features, embedding_dims, lstm_units)
        root2 = tf.train.Checkpoint(optimizer=optimizer,
                                    model=comb_adv_model2,
                                    optimizer_step=tf.optimizers.SGD())
        root2.restore(tf.train.latest_checkpoint("saved/advM2/"))

    # 3.2 test the model on original test data
    evaluate_acc(comb_adv_model, x_test, test_labels, BATCH_SIZE)
    if multitaskMode:
        evaluate_acc(comb_adv_model2, x_test, test_labels, BATCH_SIZE)

    # 3.3 test HotFlip Attack on test data
    max_perturbed = 5
    x = np.array(x_test[2:4])
    y = np.array(test_labels[2:4])
    x = x.reshape((-1, 80))
    # ay = tf.reshape(y, [-1])
    for idx, (_x, _y) in enumerate(zip(x, y)):
        print("Initial sentence:")
        print(decode_review(_x))
        _x = _x.reshape((1, 80))
        _ay = tf.reshape(_y, [-1])

        with tf.GradientTape() as tape:
            output, same = get_pred(comb_adv_model, _x, _y)
            if multitaskMode:
                output2, same2 = get_pred(comb_adv_model2, _x, _y)
                same = same or same2
        count = 0
        if same:
            while same:
                print("attack # {} ---------------------------- ".format(count))
                count += 1
                with tf.GradientTape(persistent=True) as tape:
                    _x = _x.reshape((1, 80))
                    output, same = get_pred(comb_adv_model, _x, _y)
                    loss = loss_function(_ay, output)
                    variables = comb_adv_model.trainable_variables
                    gradients = tape.gradient(loss, [variables[0]])

                    if multitaskMode:
                        output2, same2 = get_pred(comb_adv_model2, _x, _y)
                        loss2 = loss_function(_ay, output2)
                        variables2 = comb_adv_model2.trainable_variables
                        gradients2 = tape.gradient(loss2, [variables2[0]])
                        candidates, x2, which_token = hotflip_attack_multitask(_x, [gradients[0], gradients2[0]],
                                                                     mask=np.ones_like(_x[0]), model=comb_adv_model)
                        same = same or same2
                    else:
                        candidates, x2, which_token = hotflip_attack(_x, gradients[0],
                                                                     mask=np.ones_like(_x[0]), model=comb_adv_model)

                _x = x2
                print("Perturbed Sentences: {}".format(decode_review(x2)))
                print(decode_review(candidates.numpy()[0]))
                print(candidates.numpy()[0])
                if count == max_perturbed:
                    print("attack fail!")
                    break
            else:
                print("attack succeed!")
        else:
            print("wrong prediction!")