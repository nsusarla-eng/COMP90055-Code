from classifier_test import train as train_model, iterate_minibatches, load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
import numpy as np
import theano.tensor as T
import lasagne
import theano
import argparse
import os
import matplotlib.pyplot as plt
import random as r
import pandas as pd

np.random.seed(21312)
MODEL_PATH = './model/'
DATA_PATH = './data/'
RESULT_PATH = './results/'
PLOT_PATH = './plotdata/'
global exprfile

import theano.gof.compiledir as cd
cd.print_compiledir_content()

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)


def load_trained_indices():
    fname = MODEL_PATH + 'data_indices.npz'
    with np.load(fname) as f:
        indices = [f['arr_%d' % i] for i in range(len(f.files))]
    return indices


def get_data_indices(data_size, target_train_size=int(1e4), sample_target_data=True):
    train_indices = np.arange(data_size)
    if sample_target_data:
        target_data_indices = np.random.choice(train_indices, target_train_size, replace=False)
        shadow_indices = np.setdiff1d(train_indices, target_data_indices)
    else:
        target_data_indices = train_indices[:target_train_size]
        shadow_indices = train_indices[target_train_size:]
    return target_data_indices, shadow_indices


def load_attack_data():
    fname = MODEL_PATH + 'attack_train_data.npz'
    with np.load(fname) as f:
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
    fname = MODEL_PATH + 'attack_test_data.npz'
    with np.load(fname) as f:
        test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x.astype('float32'), train_y.astype('int32'), test_x.astype('float32'), test_y.astype('int32')


def train_target_model(dataset, epochs=100, batch_size=100, learning_rate=0.01, l2_ratio=1e-7,
                       n_hidden=50, model='nn', save=True):
    train_x, train_y, test_x, test_y = dataset
    print("feats:training data for target model {}".format(len(train_x)))
    print("labels:traingin data for target model {}".format(len(train_y)))
    print("feats:test data for target model {}".format(len(test_x)))
    print("labels:test data for target model {}".format(len(test_y)))
    
    
    output_layer = train_model(dataset, exprfile, n_hidden=n_hidden, epochs=epochs, learning_rate=learning_rate,
                               batch_size=batch_size, model=model, l2_ratio=l2_ratio)
    # test data for attack model
    attack_x, attack_y = [], []
    input_var = T.matrix('x')
    prob = lasagne.layers.get_output(output_layer, input_var, deterministic=True)
    prob_fn = theano.function([input_var], prob)
    
    # data used in training, label is 1
    for batch in iterate_minibatches(train_x, train_y, batch_size, False):
        attack_x.append(prob_fn(batch[0]))
        attack_y.append(np.ones(batch_size))
        
    # data not used in training, label is 0
    for batch in iterate_minibatches(test_x, test_y, batch_size, False):
        attack_x.append(prob_fn(batch[0]))
        attack_y.append(np.zeros(batch_size))

    print('test data for attack model {}'.format(len(attack_x)))
    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')

    if save:
        np.savez(MODEL_PATH + 'attack_test_data.npz', attack_x, attack_y)
        np.savez(MODEL_PATH + 'target_model.npz', *lasagne.layers.get_all_param_values(output_layer))

    classes = np.concatenate([train_y, test_y])
    
    
    return attack_x, attack_y, classes


def train_shadow_models(n_hidden=50, epochs=100, n_shadow=20, learning_rate=0.05, batch_size=100, l2_ratio=1e-7,
                        model='nn', save=True):
    # for getting probabilities
    input_var = T.matrix('x')
    # for attack model
    attack_x, attack_y = [], []
    classes = []
    for i in range(n_shadow):
        print ('Training shadow model {}'.format(i),file=exprfile)
        data = load_data('shadow{}_data.npz'.format(i))
        train_x, train_y, test_x, test_y = data
        # train model
        output_layer = train_model(data, exprfile, n_hidden=n_hidden, epochs=epochs, learning_rate=learning_rate,
                                   batch_size=batch_size, model=model, l2_ratio=l2_ratio)
        prob = lasagne.layers.get_output(output_layer, input_var, deterministic=True)
        prob_fn = theano.function([input_var], prob)
        print ('Gather training data for attack model',file=exprfile)
        attack_i_x, attack_i_y = [], []
        # data used in training, label is 1
        for batch in iterate_minibatches(train_x, train_y, batch_size, False):
            attack_i_x.append(prob_fn(batch[0]))
            attack_i_y.append(np.ones(batch_size))
        # data not used in training, label is 0
        for batch in iterate_minibatches(test_x, test_y, batch_size, False):
            attack_i_x.append(prob_fn(batch[0]))
            attack_i_y.append(np.zeros(batch_size))
        attack_x += attack_i_x
        attack_y += attack_i_y
        classes.append(np.concatenate([train_y, test_y]))
    # train data for attack model
    print('training data for attack model {}'.format(len(attack_x)))
    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')
    classes = np.concatenate(classes)
    if save:
        np.savez(MODEL_PATH + 'attack_train_data.npz', attack_x, attack_y)

    return attack_x, attack_y, classes


def train_attack_model(classes, dataset=None, n_hidden=50, learning_rate=0.01, batch_size=200, epochs=50,
                       model='nn', l2_ratio=1e-7):
    
    if dataset is None:
        dataset = load_attack_data()

    train_x, train_y, test_x, test_y = dataset
        
    train_classes, test_classes = classes
    train_indices = np.arange(len(train_x))
    test_indices = np.arange(len(test_x))
    unique_classes = np.unique(train_classes)

    true_y = []
    pred_y = []
    #class_accry = np.zeros((len(unique_classes),2))
    class_accry = []
    class_pres_recall =[]
    for c in unique_classes:
        print ('Training attack model for class {}...'.format(c),file=exprfile)
        print(model)
        c_train_indices = train_indices[train_classes == c]
        c_train_x, c_train_y = train_x[c_train_indices], train_y[c_train_indices]
        c_test_indices = test_indices[test_classes == c]
        c_test_x, c_test_y = test_x[c_test_indices], test_y[c_test_indices]
        c_dataset = (c_train_x, c_train_y, c_test_x, c_test_y)
        c_pred_y = train_model(c_dataset, exprfile, n_hidden=n_hidden, epochs=epochs, learning_rate=learning_rate,
                               batch_size=batch_size, model=model, rtn_layer=False, l2_ratio=l2_ratio)
        true_y.append(c_test_y)
        pred_y.append(c_pred_y)
        
        # class_pres_recall.append(accuracy_score(c_test_y, c_pred_y))
        # class_pres_recall.append(precision_score(c_test_y, c_pred_y)) #of the positive class
        # class_pres_recall.append(recall_score(c_test_y, c_pred_y))
        # class_pres_recall.append(precision_score(c_test_y, c_pred_y, average='weighted'))
        # class_pres_recall.append(recall_score(c_test_y, c_pred_y, average='weighted'))
        
    
    # class_pres_recall = np.asarray(class_pres_recall)
    # class_pres_recall = np.reshape(class_pres_recall, (len(unique_classes),5))
    
    # np.savetxt(PLOT_PATH + 'Texas90Knowledge_Attack.csv', class_pres_recall, delimiter=',')
    
    print ('-' * 10 + 'FINAL EVALUATION' + '-' * 10 + '\n',file=exprfile)
    true_y = np.concatenate(true_y)
    pred_y = np.concatenate(pred_y)
    print ('Testing Accuracy: {}'.format(accuracy_score(true_y, pred_y)),file=exprfile)
    print (classification_report(true_y, pred_y),file=exprfile)

   
def save_data():
    print ('-' * 10 + 'SAVING DATA TO DISK' + '-' * 10 + '\n',file=exprfile)

    x, y, test_x, test_y = load_dataset(args.train_feat, args.train_label, args.test_feat, args.train_label)
    
    
    if test_x is None:
        print ('Splitting train/test data with ratio {}/{}'.format(1 - args.test_ratio, args.test_ratio),file=exprfile)
        x, test_x, y, test_y = train_test_split(x, y, test_size=args.test_ratio, stratify=y)

    # need to partition target and shadow model data
    assert len(x) > 2 * args.target_data_size

    target_data_indices, shadow_indices = get_data_indices(len(x), target_train_size=args.target_data_size)
    np.savez(MODEL_PATH + 'data_indices.npz', target_data_indices, shadow_indices)

    # target model's data
    print ('Saving data for target model', file=exprfile)
    train_x, train_y = x[target_data_indices], y[target_data_indices]
    
        
       
    size = len(target_data_indices)
    if size < len(test_x):
        test_x = test_x[:size]
        test_y = test_y[:size]
        
        
    #Leveraging attacker knowledge for Attack model
    knowledge = train_x[:,:2469]
    print(knowledge.shape)
    knowledge_pd = pd.DataFrame(knowledge)
    knowledge_pd = knowledge_pd.drop_duplicates()
        
    test_pd = pd.DataFrame(test_x)
    test_pd_y = pd.DataFrame(test_y)
    test_pd.insert(6169, "Label", test_pd_y, True) #this was 8 for Adult dataset
    print(test_pd.shape)
    match = test_pd.merge(knowledge_pd, on=(list(knowledge_pd.columns)), how='inner')
    print(match.shape)
    print(match)
    match['key'] = 'x'
    temp_df = test_pd.merge(match, on=test_pd.columns.tolist(), how='left')
    nomatch = temp_df[temp_df['key'].isnull()].drop('key', axis=1)
    print(nomatch.shape)
    test_pd_y = match['Label']
    df = match.drop(['Label', 'key'], axis=1)
    print(df.head(10))
    test_y = test_pd_y.to_numpy()
    test_x = df.to_numpy()
    print(test_y.shape)
    print(test_x.shape)
    
    # save target data
    np.savez(DATA_PATH + 'target_data.npz', train_x, train_y, test_x, test_y)

    # shadow model's data
    target_size = len(target_data_indices)
    shadow_x, shadow_y = x[shadow_indices], y[shadow_indices]
    shadow_indices = np.arange(len(shadow_indices))

    for i in range(args.n_shadow):
        print ('Saving data for shadow model {}'.format(i), file=exprfile)
        shadow_i_indices = np.random.choice(shadow_indices, 2 * target_size, replace=False)
        shadow_i_x, shadow_i_y = shadow_x[shadow_i_indices], shadow_y[shadow_i_indices]
        train_x, train_y = shadow_i_x[:target_size], shadow_i_y[:target_size]
        #shadow_train_x, shadow_train_y = shadow_i_x[:target_size], shadow_i_y[:target_size]
        #shadow_train_x = np.append(train_x[:,:2467],shadow_train_x[:,2467:],axis=1)
        #shadow_train_y = np.append(train_y[:2467], shadow_train_y[2467:])
        
        #Code for the filling in the training data with data from a larger dataset
        #train_x = np.append(train_x[:,:5], shadow_i_x[:target_size,5:],axis=1)
        test_x, test_y = shadow_i_x[target_size:], shadow_i_y[target_size:]
        
        #Code for binary random numbers filling in the remaining dataset
        # random_data = np.random.randint(0,2,shadow_i_x.shape)
        # train_x = np.append(train_x[:,:2467], random_data[:target_size,2467:],axis=1)
        # test_x, test_y = shadow_i_x[target_size:], shadow_i_y[target_size:]
        #test_x = np.append(shadow_i_x[target_size:,:5552], random_data[target_size:,5552:],axis=1)
        
        #Code for binary filling for Adult UCI
        
        # columns_feat1 = 5
        # columns_feat2 = 8
        # columns_feat3 = 1
        # columns_feat4 = 2
        # random_data1 = np.zeros((target_size,columns_feat1))
        # random_data2 = np.zeros((target_size,columns_feat2))
        # random_data3 = np.zeros((target_size,columns_feat3))
        # random_data4 = np.zeros((target_size,columns_feat4))

        # for row in range(target_size):
        #     x = r.randint(0, columns_feat1-1)
        #     y = r.randint(0, columns_feat2-1)
        #     z = r.randint(30,50)
        #     a = r.randint(0, columns_feat4-1)
        #     random_data1[row][x] = 1
        #     random_data2[row][y] = 1
        #     random_data3[row] = z
        #     random_data4[row][a] = 1
        
        
        # temp1 = np.append(random_data2, random_data1,axis=1)
        # temp2 = np.append(random_data3, temp1,axis=1)
        # random_data = np.append(random_data4, temp2,axis=1)
        # new_train_x = np.append(train_x[:,:12], random_data,axis=1)
        # print(new_train_x.shape)
        # train_x = new_train_x
        # test_x, test_y = shadow_i_x[target_size:], shadow_i_y[target_size:]
        
        
        np.savez(DATA_PATH + 'shadow{}_data.npz'.format(i), train_x, train_y, test_x, test_y)


def load_data(data_name):
    print(DATA_PATH + data_name, file=exprfile)
    with np.load(DATA_PATH + data_name) as f:
        train_x, train_y, test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x, train_y, test_x, test_y


def attack_experiment():
    print ('-' * 10 + 'TRAIN TARGET' + '-' * 10 + '\n', file=exprfile)
    dataset = load_data('target_data.npz')
    attack_test_x, attack_test_y, test_classes = train_target_model(
        dataset=dataset,
        epochs=args.target_epochs,
        batch_size=args.target_batch_size,
        learning_rate=args.target_learning_rate,
        n_hidden=args.target_n_hidden,
        l2_ratio=args.target_l2_ratio,
        model=args.target_model,
        save=args.save_model)

    print ('-' * 10 + 'TRAIN SHADOW' + '-' * 10 + '\n', file=exprfile)
    attack_train_x, attack_train_y, train_classes = train_shadow_models(
        epochs=args.target_epochs,
        batch_size=args.target_batch_size,
        learning_rate=args.target_learning_rate,
        n_shadow=args.n_shadow,
        n_hidden=args.target_n_hidden,
        l2_ratio=args.target_l2_ratio,
        model=args.target_model,
        save=args.save_model)

    print ('-' * 10 + 'TRAIN ATTACK' + '-' * 10 + '\n', file=exprfile)
    dataset = (attack_train_x, attack_train_y, attack_test_x, attack_test_y)
    train_attack_model(
        dataset=dataset,
        epochs=args.attack_epochs,
        batch_size=args.attack_batch_size,
        learning_rate=args.attack_learning_rate,
        n_hidden=args.attack_n_hidden,
        l2_ratio=args.attack_l2_ratio,
        model=args.attack_model,
        classes=(train_classes, test_classes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_feat', type=str)
    parser.add_argument('train_label', type=str)
    parser.add_argument('--test_feat', type=str, default=None)
    parser.add_argument('--test_label', type=str, default=None)
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--save_data', type=int, default=0)
    # if test not give, train test split configuration
    parser.add_argument('--test_ratio', type=float, default=0.3)
    # target and shadow model configuration
    parser.add_argument('--n_shadow', type=int, default=10)
    parser.add_argument('--target_data_size', type=int, default=int(1e4))   # number of data point used in target model
    parser.add_argument('--target_model', type=str, default='nn')           # FNB: this flag specifies the type of ML used. i.e. softmax or nn
    parser.add_argument('--target_learning_rate', type=float, default=0.01)
    parser.add_argument('--target_batch_size', type=int, default=100)
    parser.add_argument('--target_n_hidden', type=int, default=50)
    parser.add_argument('--target_epochs', type=int, default=50)
    parser.add_argument('--target_l2_ratio', type=float, default=1e-6)
    parser.add_argument('--attacker_know', type=float, default=0)
    # attack model configuration
    parser.add_argument('--attack_model', type=str, default='softmax')
    parser.add_argument('--attack_learning_rate', type=float, default=0.01)
    parser.add_argument('--attack_batch_size', type=int, default=100)
    parser.add_argument('--attack_n_hidden', type=int, default=50)
    parser.add_argument('--attack_epochs', type=int, default=50)
    parser.add_argument('--attack_l2_ratio', type=float, default=1e-6)

    # parse configuration
    args = parser.parse_args()
    exprfile = open(RESULT_PATH + 'Texas40Knowledge_Attack.txt','a')
    print (vars(args), file=exprfile)
    print('Experimental setup: Texas data with attack model changes - 40% feature knowledge of training data', file=exprfile)
    
    if args.save_data:
        save_data()
        exprfile.close()
    else:
        attack_experiment()
        exprfile.close()
