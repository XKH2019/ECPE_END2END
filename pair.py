# encoding: utf-8
# @author: zxding
# email: d.z.x@qq.com


import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
import sys, os, time, codecs, pdb

from utils.tf_funcs import *
from utils.prepare_data import *
from layer import ECFU

FLAGS = tf.app.flags.FLAGS
# >>>>>>>>>>>>>>>>>>>> For Model <<<<<<<<<<<<<<<<<<<< #
## embedding parameters ##
tf.app.flags.DEFINE_string('w2v_file', '../data/w2v_200 .txt', 'embedding file')
tf.app.flags.DEFINE_integer('embedding_dim', 200, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('embedding_dim_pos', 50, 'dimension of position embedding')
## input struct ##
tf.app.flags.DEFINE_integer('max_sen_len', 30, 'max number of tokens per sentence')
## model struct ##
tf.app.flags.DEFINE_integer('n_hidden', 100, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 2, 'number of distinct class')
# >>>>>>>>>>>>>>>>>>>> For Data <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_string('log_file_name', '', 'name of log file')
# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_integer('training_iter', 10, 'number of train iter')
tf.app.flags.DEFINE_string('scope', 'P_cause', 'RNN scope')
# not easy to tune , a good posture of using data to train model is very important
tf.app.flags.DEFINE_integer('batch_size', 32, 'number of example per batch')
tf.app.flags.DEFINE_float('learning_rate', 0.005, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob1', 0.5, 'word embedding training dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'softmax layer dropout keep prob')
tf.app.flags.DEFINE_float('l2_reg', 0.00001, 'l2 regularization')


            #       语料库中的词向量 位置向量   输入 子句长度                         距离word_embedding, pos_embedding, x, sen_len, x_context,x_context_len,keep_prob1, keep_prob2, distance, y
def build_model(word_embedding, pos_embedding, x, sen_len, x_context,x_context_len,keep_prob1, keep_prob2, distance, y,doc_len_context, RNN = biLSTM):
    # 上下文
    x_context = tf.nn.embedding_lookup(word_embedding, x_context)
    inputs = tf.reshape(x_context, [-1, FLAGS.max_sen_len, FLAGS.embedding_dim])  # 沿着75展开   30 200
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)
    x_context_len = tf.reshape(x_context_len, [-1])  # batchsize 个子句长度

    def get_s(inputs, name):  # xianzai的inputs 不是文档级 batchsize级别的
        with tf.name_scope('word_encode'):
            inputs = RNN(inputs, x_context_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + 'word_layer' + name)  # 30 200
        with tf.name_scope('word_attention'):
            sh2 = 2 * FLAGS.n_hidden
            w1 = get_weight_varible('word_att_w1' + name, [sh2, sh2])
            b1 = get_weight_varible('word_att_b1' + name, [sh2])
            w2 = get_weight_varible('word_att_w2' + name, [sh2, 1])
            s = att_var(inputs, x_context_len, w1, b1, w2)  # (?,200)
        s = tf.reshape(s, [-1, FLAGS.max_doc_len, 2 * FLAGS.n_hidden])  # (?,75,200)
        return s

    xx = get_s(inputs, name='cause_word_encode')  # s  30 200
    xx = RNN(xx, doc_len_context, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + 'cause_sentence_layer')   #30 200
    # 情感原因句对
    x = tf.nn.embedding_lookup(word_embedding, x)
    x_inputs = tf.reshape(x, [-1, FLAGS.max_sen_len, FLAGS.embedding_dim])  #最后一维每个词
    x_inputs = tf.nn.dropout(x_inputs, keep_prob=keep_prob1)   #防止过拟合 （30，500）
    sen_len = tf.reshape(sen_len, [-1])     #将每个子句长度展成一维，然后进行顺序输入
    # def x_get_s(inputs, name):
    #     with tf.name_scope('word_encode'):
    #         inputs = RNN(inputs, sen_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope+'word_layer' + name)    #[-1,maxlen,hidden*2]
    #
    #     with tf.name_scope('word_attention'):
    #         sh2 = 2 * FLAGS.n_hidden
    #         w1 = get_weight_varible('word_att_w1' + name, [sh2, sh2])
    #         b1 = get_weight_varible('word_att_b1' + name, [sh2])
    #         w2 = get_weight_varible('word_att_w2' + name, [sh2, 1])
    #         s = att_var(inputs,sen_len,w1,b1,w2)
    #     s = tf.reshape(s, [-1, 2 * 2 * FLAGS.n_hidden])
    #     return s

    def x_get_s(inputs, name):
        inputs = RNN(inputs, sen_len, n_hidden=FLAGS.n_hidden,scope=FLAGS.scope + 'word_layer' + name)  # [-1,maxlen,hidden*2]
        return inputs
    s = x_get_s(x_inputs, name='cause_word_encode')   #2
    s=tf.reshape(s,[2,FLAGS.batch_size,FLAGS.max_sen_len, FLAGS.embedding_dim])

    #句对之间距离
    dis = tf.nn.embedding_lookup(pos_embedding, distance)
    # 融合
    ecfu=ECFU(bs=FLAGS.batch_size, sent_len=FLAGS.max_sen_len, n_in=2*FLAGS.n_hidden, n_out=2*FLAGS.n_hidden, name="ECFU")
    #两层
    h1=ecfu(xx,s)

    h2=ecfu(h1,s)
    # 拼接
    s = tf.concat([h2, dis], 1)

    s1 = tf.nn.dropout(s, keep_prob=keep_prob2)
    w_pair = get_weight_varible('softmax_w_pair', [FLAGS.max_sen_len * FLAGS.embedding_dim + FLAGS.embedding_dim_pos, FLAGS.n_class])
    b_pair = get_weight_varible('softmax_b_pair', [FLAGS.n_class])
    pred_pair = tf.nn.softmax(tf.matmul(s1, w_pair) + b_pair)      #NX2  实际概率
        
    reg = tf.nn.l2_loss(w_pair) + tf.nn.l2_loss(b_pair)       #biaoliang
    return pred_pair, reg

def print_training_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:\n')
    print('batch-{}, lr-{}, kb1-{}, kb2-{}, l2_reg-{}'.format(
        FLAGS.batch_size,  FLAGS.learning_rate, FLAGS.keep_prob1, FLAGS.keep_prob2, FLAGS.l2_reg))
    print('training_iter-{}, scope-{}\n'.format(FLAGS.training_iter, FLAGS.scope))

def get_batch_data(x, sen_len,x_context,x_context_len, keep_prob1, keep_prob2, distance, y,doc_len_context, batch_size, test=False):
    for index in batch_index(len(y), batch_size, test):
        feed_list = [x[index], sen_len[index],x_context[index],x_context_len[index], keep_prob1, keep_prob2, distance[index], y[index],doc_len_context[index]]
        yield feed_list, len(index)

def run():
    save_dir = 'pair_data/{}/'.format(FLAGS.scope)
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    if FLAGS.log_file_name:          #log  file name
        sys.stdout = open(save_dir + FLAGS.log_file_name, 'w')
    print_time()
    tf.reset_default_graph()
    # Model Code Block    数据集中的词                                                     200                50                                数据集               词向量文件
    word_idx_rev, word_id_mapping, word_embedding, pos_embedding = load_w2v(FLAGS.embedding_dim, FLAGS.embedding_dim_pos, 'data_combine/clause_keywords.csv', FLAGS.w2v_file)
    word_embedding = tf.constant(word_embedding, dtype=tf.float32, name='word_embedding')
    pos_embedding = tf.constant(pos_embedding, dtype=tf.float32, name='pos_embedding')       #位置向量是一个高斯分布

    print('build model...')

    #定义输入（每个pair：2行30列） 句子长度（n行2列）      距离
    x = tf.placeholder(tf.int32, [None, 2, FLAGS.max_sen_len])          #一个bachsize是送多少个pair
    sen_len = tf.placeholder(tf.int32, [None, 2])
    keep_prob1 = tf.placeholder(tf.float32)      #标量
    keep_prob2 = tf.placeholder(tf.float32)
    distance = tf.placeholder(tf.int32, [None])  #每个pair的一个距离
    y = tf.placeholder(tf.float32, [None, FLAGS.n_class]) #每个pair标签两列 （1，0）（0，1）
    x_context=tf.placeholder(tf.int32, [None, 30, FLAGS.max_sen_len])
    x_context_len=tf.placeholder(tf.int32, [None, 30])
    doc_len_context=tf.placeholder(tf.int32, [None])
    placeholders = [x, sen_len, x_context,x_context_len,keep_prob1, keep_prob2, distance, y,doc_len_context]
    
    #N*2 预测的正确的         标量
    pred_pair, reg = build_model(word_embedding, pos_embedding, x, sen_len, x_context,x_context_len,keep_prob1, keep_prob2, distance, y,doc_len_context)
    loss_op = - tf.reduce_mean(y * tf.log(pred_pair)) + reg * FLAGS.l2_reg
    #训练有而测试没有
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss_op)
    
    true_y_op = tf.argmax(y, 1)         #真正正确的      输出下标
    pred_y_op = tf.argmax(pred_pair, 1)   #预测正确的
    acc_op = tf.reduce_mean(tf.cast(tf.equal(true_y_op, pred_y_op), tf.float32))   #准确率用预测正确的与真正正确的个数除以总个数
    print('build model done!\n')
    
    # Training Code Block
    print_training_info()
    tf_config = tf.ConfigProto()  
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        keep_rate_list, acc_subtask_list, p_pair_list, r_pair_list, f1_pair_list = [], [], [], [], []
        o_p_pair_list, o_r_pair_list, o_f1_pair_list = [], [], []
        
        for fold in range(1,11):
            sess.run(tf.global_variables_initializer())
            # train for one fold
            print('############# fold {} begin ###############'.format(fold))
            # Data Code Block
            train_file_name = 'fold{}_train.txt'.format(fold, FLAGS)
            test_file_name = 'fold{}_test.txt'.format(fold)
            tr_pair_id_all, tr_pair_id, tr_y, tr_x, tr_sen_len, tr_distance,tr_x_context,tr_x_context_len ,tr_doc_len_context= load_data_2nd_step(save_dir + train_file_name, word_id_mapping, max_sen_len = FLAGS.max_sen_len)
            te_pair_id_all, te_pair_id, te_y, te_x, te_sen_len, te_distance,te_x_context,te_x_context_len ,te_doc_len_context= load_data_2nd_step(save_dir + test_file_name, word_id_mapping, max_sen_len = FLAGS.max_sen_len)
            
            max_acc_subtask, max_f1 = [-1.]*2   #最大的准确率 和最大的f1值 每一个文件
            print('train docs: {}    test docs: {}'.format(len(tr_x), len(te_x)))

            for i in range(FLAGS.training_iter):
                start_time, step = time.time(), 1
                # train   yige epoch
                for train, _ in get_batch_data(tr_x, tr_sen_len,tr_x_context,tr_x_context_len, FLAGS.keep_prob1, FLAGS.keep_prob2, tr_distance, tr_y, tr_doc_len_context,FLAGS.batch_size):  #get_batch_size解决！
                    _, loss, pred_y, true_y, acc = sess.run(
                        [optimizer, loss_op, pred_y_op, true_y_op, acc_op], feed_dict=dict(zip(placeholders, train)))
                    print('step {}: train loss {:.4f} acc {:.4f}'.format(step, loss, acc))
                    step = step + 1



                # test直接在整个测试集上进行

                test = [te_x, te_sen_len, 1., 1., te_distance, te_y]
                loss, pred_y, true_y, acc = sess.run([loss_op, pred_y_op, true_y_op, acc_op], feed_dict=dict(zip(placeholders, test)))
                print('\nepoch {}: test loss {:.4f}, acc {:.4f}, cost time: {:.1f}s\n'.format(i, loss, acc, time.time()-start_time))

                if acc > max_acc_subtask:
                    max_acc_subtask = acc           #每个epoch最大的acc
                print('max_acc_subtask: {:.4f} \n'.format(max_acc_subtask))
                
                # p, r, f1, o_p, o_r, o_f1, keep_rate = prf_2nd_step(te_pair_id_all, te_pair_id, pred_y, fold, save_dir)
                p, r, f1, o_p, o_r, o_f1, keep_rate = prf_2nd_step(te_pair_id_all, te_pair_id, pred_y)
                if f1 > max_f1:
                    max_keep_rate, max_p, max_r, max_f1 = keep_rate, p, r, f1
                print('original o_p {:.4f} o_r {:.4f} o_f1 {:.4f}'.format(o_p, o_r, o_f1))
                print ('pair filter keep rate: {}'.format(keep_rate))
                print('test p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1))    #输出最大的p r f

                print('max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(max_p, max_r, max_f1))   #输出最大的p r f
                

            print( 'Optimization Finished!\n')
            print('############# fold {} end ###############'.format(fold))
            # fold += 1  15个epoch完成了
            acc_subtask_list.append(max_acc_subtask)          #tian加每个fold的最大的 准确率
            keep_rate_list.append(max_keep_rate)
            p_pair_list.append(max_p)
            r_pair_list.append(max_r)
            f1_pair_list.append(max_f1)
            o_p_pair_list.append(o_p)
            o_r_pair_list.append(o_r)
            o_f1_pair_list.append(o_f1)                  #10个
            
              
        print_training_info()
        all_results = [acc_subtask_list, keep_rate_list, p_pair_list, r_pair_list, f1_pair_list, o_p_pair_list, o_r_pair_list, o_f1_pair_list]
        acc_subtask, keep_rate, p_pair, r_pair, f1_pair, o_p_pair, o_r_pair, o_f1_pair = map(lambda x: np.array(x).mean(), all_results)
        print('\nOriginal pair_predict: test f1 in 10 fold: {}'.format(np.array(o_f1_pair_list).reshape(-1,1)))
        print('average yuanshi: p {:.4f} r {:.4f} f1 {:.4f}\n'.format(o_p_pair, o_r_pair, o_f1_pair))
        print('\nAverage keep_rate: {:.4f}\n'.format(keep_rate))
        print('\nFiltered pair_predict: test f1 in 10 fold: {}'.format(np.array(f1_pair_list).reshape(-1,1)))
        print('average : p {:.4f} r {:.4f} f1 {:.4f}\n'.format(p_pair, r_pair, f1_pair))
        print_time()
        
     
def main(_):
    
    # FLAGS.log_file_name = 'step2.log'
    FLAGS.training_iter=20

    for scope_name in ['Ind_BiLSTM', 'P_emotion', 'P_cause']:
        FLAGS.scope= scope_name + '_1'
        run()
        FLAGS.scope= scope_name + '_2'
        run()

    


if __name__ == '__main__':
    tf.app.run() 