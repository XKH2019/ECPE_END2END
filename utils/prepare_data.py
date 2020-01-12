# encoding:utf-8

import codecs
import random
import numpy as np
import pickle as pk
from sklearn.metrics import precision_score, recall_score, f1_score
import pdb, time

def print_time():
    print('\n----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))

def load_w2v(embedding_dim, embedding_dim_pos, train_file_path, embedding_path):
    print('\nload embedding...')

    words = []
    inputFile1 = open(train_file_path, 'r',encoding='utf-8')
    for line in inputFile1.readlines():
        line = line.strip().split(',')
        emotion, clause = line[2], line[-1]     #情感词和子句
        words.extend( [emotion] + clause.split())
    words = set(words)  # 所有不重复词的集合  建立词库
   # print(words)
    word_idx = dict((c, k + 1) for k, c in enumerate(words)) # 每个词及词的位置 下标从1开始
    word_idx_rev = dict((k + 1, c) for k, c in enumerate(words)) # 每个词的序号及词 是不是为了找词向量
    #建立词库完成！建立词序号完成!
    w2v = {}                            #词向量
    inputFile2 = open(embedding_path, 'r',encoding='utf-8')
    inputFile2.readline()
    for line in inputFile2.readlines():
        line = line.strip().split(' ')
        w, ebd = line[0], line[1:]      #liebiao
        w2v[w] = ebd        #词向量字典wenjain

    embedding = [list(np.zeros(embedding_dim))]   #第一行是0向量
    hit = 0        #语料库中在词向量文件中的词的个数
    for item in words:    #语料库中的词
        if item in w2v:   #词向量文件中的词
            vec = list(map(float, w2v[item]))
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1) # 从均匀分布[,0.1]中随机取  如果不在词向量库中，则随机生成
        embedding.append(vec)


        #数据集的ebedding构造完成!

    print('w2v_file: {}\nall_words: {} hit_words: {}'.format(embedding_path, len(words), hit))

    embedding_pos = [list(np.zeros(embedding_dim_pos))]       #第一行是0   标准差0.1的高斯分布
    embedding_pos.extend( [list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(200)] )      #随机生成200个位置向量

    embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)
    
    print("embedding.shape: {} embedding_pos.shape: {}".format(embedding.shape, embedding_pos.shape))
    print("load embedding done!\n")
    return word_idx_rev, word_idx, embedding, embedding_pos


def load_data(input_file, word_idx, max_doc_len = 30, max_sen_len =20 ):
    print('load data_file: {}'.format(input_file))
    distance,  y, x, sen_len, doc_len,senctx,senctx_len ,pair_id_all,y_pairs= [], [], [], [], [], [],[],[],[]
    doc_id = []
    
    n_cut = 0
    inputFile = open(input_file, 'r',encoding='utf-8')
    while True:
        sentens=[]
        lenth=0
        line = inputFile.readline()
        if line == '': break
        line = line.strip().split()
        doc_id.append(line[0])    #文档id
        d_len = int(line[1])        #文档长度
        pairs = eval('[' + inputFile.readline().strip() + ']') #执行字符串表达式
        doc_len.append(d_len)     #第一个文档子句个数
        y_pairs.append(pairs)     #第一个文档的子句对数
        # pos, cause = zip(*pairs)    #第一个文档的情感句和原因句  反zip 返回一个列表
        pair_id_all.extend([doc_id*10000+p[0]*100+p[1] for p in pairs])   #添加每个文档的预测的pair的权重化值
        # 分配最大的矩阵空间，其余部分是0
        sen_len_tmp, x_tmp = np.zeros(max_doc_len,dtype=np.int32), np.zeros((max_doc_len, max_sen_len),dtype=np.int32)
        for i in range(d_len):
            # y_po[i][int(i+1 in pos)]=1
            # y_ca[i][int(i+1 in cause)]=1
            words = inputFile.readline().strip().split(',')[-1]
            words=words.split()
            sen_len_tmp[i] = min(len(words), max_sen_len)   #zijiu的真实长度  75长数组

            lenth += sen_len_tmp[i]
            for j in range(sen_len_tmp[i]):
                if len(sentens) >200:
                    break
                sentens.append(words[j])

            for j, word in enumerate(words):
                if j >= max_sen_len:
                    n_cut += 1                                  #单词数量
                    break
                x_tmp[i][j] = int(word_idx[word])       #将文档将子句装换成索引

        # y_position.append(y_po)
        # y_cause.append(y_ca)
        x.append(x_tmp)
        sen_len.append(sen_len_tmp)        #每个文档子句的长度
        senctx.append(sentens)
        senctx_len.append(lenth)
        if d_len>30:d_len=30
        for i in range(d_len):
            for j in range(d_len):
                pair_id_cur = doc_id*10000+i*100+j
                y.append([0,1] if pair_id_cur in pair_id_all else [1,0])   #如果预测正确 则添加【0，1】 y中包括所有预测的pair    创造的真是标签
                distance.append(j-i+100)        #加100？


    x, sen_len, doc_len,senctx,senctx_len ,y,distance,doc_id= map(np.array, [x, sen_len, doc_len,senctx,senctx_len,y,distance,doc_id])
    for var in ['y_position', 'y_cause', 'x', 'sen_len', 'doc_len']:
        print('{}.shape {}'.format( var, eval(var).shape ))
    print('n_cut {}'.format(n_cut))
    print('load data done!\n')
    return x, sen_len, doc_len,senctx,senctx_len ,y,distance,doc_id


def stopwordslist():
     stopwords = [line.strip() for line in open('data/stopwords.txt',encoding='UTF-8').readlines()]
     return stopwords

def load_data_2nd_step(input_file, word_idx, max_doc_len = 75, max_sen_len = 30):
   #  print('load data_file: {}'.format(input_file))
   #  pair_id_all, pair_id, y, xc1,xc2,x, sen_len, distance,doclongth = [],[], [], [], [], [], [],[],[]
   #  originsenlen=[]  #meige zi句真实长度 用于计算 每个子句中词的到子句的距离
   #  n_cut = 0
   #  inputFile = open(input_file, 'r')
   #  while True:
   #      long=0
   #      sentense=[]
   #      line = inputFile.readline()
   #      if line == '': break
   #      line = line.strip().split()
   #      doc_id = int(line[0])
   #      d_len = int(line[1])
   #      pairs = list(eval(inputFile.readline().strip()))    #
   #
   #      pair_id_all.extend([doc_id*10000+p[0]*100+p[1] for p in pairs])   #真实标签
   #      sen_len_tmp, x_tmp = np.zeros(max_doc_len,dtype=np.int32), np.zeros((max_doc_len, max_sen_len),dtype=np.int32)  #（75） ，（75，30）
   #      # pos_list, cause_list = [], []
   #      # originsen=[]
   #      for i in range(d_len):
   #          # senlong=0
   #          # for j in range(d_len):
   #          #读取每个子句
   #          line = inputFile.readline().strip().split(',')
   #          # if int(line[1].strip())>0:
   #          #     pos_list.append(i+1)
   #          # if int(line[2].strip())>0:
   #          #     cause_list.append(i+1)
   #          words = line[-1]      #zifuchuan
   #          # senlong=len(words.split())     #返回时列表 字符
   #
   #          # long+=senlong                   #总长度
   #          sen_len_tmp[i] = min(len(words.split()), max_sen_len)   #每个句子的实际长度
   #
   #          long+=sen_len_tmp[i]
   #
   #
   #          for j, word in enumerate(words.split()):
   #              if j >= max_sen_len:
   #                  n_cut += 1                  #切分句子的数目
   #                  break
   #              x_tmp[i][j] = int(word_idx[word])
   #          sentense.extend(x_tmp[i])
   #
   #          words=words.split()  #每个子句的列表
   #          # originsen.append(words)
   #
   #          stopwords=stopwordslist()
   #          for i in range(len(words)):
   #              if words[i] in stopwords:
   #                  del words[i]
   #
   #      originsenlen=np.array(originsenlen.append(sen_len_tmp))    #每个子句词数 N*75
   #      doclongth.append(long)   #wendang 总词数
   #      x.append(sentense)
   #
   #          #存储句子
   #      #结果 ： 真实的pair标签 预测的情感 原因标签 每个句子的实际长度 每个句子的index
   #      for i in range(d_len):   #加过1
   #          for j in range(d_len):
   #              pair_id_cur = doc_id*10000+i*100+j
   #              pair_id.append(pair_id_cur)         #预测 的pair值
   #              y.append([0,1] if pair_id_cur in pair_id_all else [1,0])   #如果预测正确 则添加【0，1】 y中包括所有预测的pair    创造的真是标签
   #              xc1.append(x_tmp[i-1])
   #              xc2.append(x_tmp[j-1])
   #              sen_len.append([sen_len_tmp[i-1], sen_len_tmp[j-1]])   #ziju实际长度
   #              distance.append(j-i+100)        #加100？子句之间距离
   # #N*2  N*2*45 N*2  N
   #  y, x, xc1,xc2,sen_len, distance,originsenlen,doclongth = map(np.array, [y, x, sen_len, distance])
   #
   #  for var in ['y', 'x', 'sen_len', 'distance']:
   #      print('{}.shape {}'.format( var, eval(var).shape ))
   #  print('n_cut {}, (y-negative, y-positive): {}'.format(n_cut, y.sum(axis=0)))
   #  print('load data done!\n')
   #
   #  return pair_id_all, pair_id, y, x, xc1,xc2,sen_len, distance,originsenlen,doclongth
   max_context_len=30
   print('load data_file: {}'.format(input_file))
   pair_id_all, pair_id, y, x, sen_len, distance ,x_context,x_context_len,doc_len_context= [], [], [], [], [], [],[],[],[]

   n_cut = 0
   inputFile = open(input_file, 'r')
   while True:
       line = inputFile.readline()  # diyihang
       if line == '': break
       line = line.strip().split()
       doc_id = int(line[0])
       d_len = int(line[1])
       if d_len < 30:
           doc_len_context.append(d_len)
       else:
           doc_len_context.append(30)

       pairs = eval(inputFile.readline().strip())  # yuce的pair列表  dierhang

       pair_id_all.extend([doc_id * 10000 + p[0] * 100 + p[1] for p in pairs])  # 添加每个文档的预测的pair的权重化值

       sen_len_tmp, x_tmp = np.zeros(max_doc_len, dtype=np.int32), np.zeros((max_doc_len, max_sen_len),
                                                                        dtype=np.int32)  # （75） ，（75，30）
       x_ctx = np.zeros((max_context_len,max_sen_len),dtype=np.int32)
       x_contextlen=np.zeros(max_context_len,dtype=np.int32)
       pos_list, cause_list = [], []
       for i in range(d_len):
           line = inputFile.readline().strip().split(',')
           if int(line[1].strip()) > 0:
               pos_list.append(i + 1)
           if int(line[2].strip()) > 0:
               cause_list.append(i + 1)
           words = line[-1]
           sen_len_tmp[i] = min(len(words.split()), max_sen_len)  # 每个句子的实际长度
           if i <max_context_len:x_contextlen[i]=min(len(words.split()),max_sen_len)

           for j, word in enumerate(words.split()):
               if j >= max_sen_len:
                   n_cut += 1  # 切分句子的数目
                   break
               x_tmp[i][j] = int(word_idx[word])  # 存储句子
               if i<max_sen_len : x_ctx[i][j]=int(word_idx[word]) #记录上下文信息
       # 结果 ： 真实的pair标签 预测的情感 原因标签 每个句子的实际长度 每个句子的index

       x_context.append(x_ctx)
       x_context_len.append(x_contextlen)

       for i in pos_list:  # 加过1
           for j in cause_list:
               pair_id_cur = doc_id * 10000 + i * 100 + j
               pair_id.append(pair_id_cur)  # 预测 的pair值

               y.append([0, 1] if pair_id_cur in pair_id_all else [1, 0])  # 如果预测正确 则添加【0，1】 y中包括所有预测的pair    创造的真是标签

               x.append([x_tmp[i - 1], x_tmp[j - 1]])

               sen_len.append([sen_len_tmp[i - 1], sen_len_tmp[j - 1]])
               distance.append(j - i + 100)  # 加100？




   # N*2  N*2*45 N*2  N
   y, x, sen_len, distance,x_context,x_context_len,doc_len_context = map(np.array, [y, x, sen_len, distance,x_context,x_context_len,doc_len_context])

   for var in ['y', 'x', 'sen_len', 'distance']:
       print('{}.shape {}'.format(var, eval(var).shape))
   print('n_cut {}, (y-negative, y-positive): {}'.format(n_cut, y.sum(axis=0)))
   print('load data done!\n')

   return pair_id_all, pair_id, y, x, sen_len, distance,x_context,x_context_len,doc_len_context


def acc_prf(pred_y, true_y, doc_len, average='binary'):
    tmp1, tmp2 = [], []
    for i in range(pred_y.shape[0]):
        for j in range(doc_len[i]):
            tmp1.append(pred_y[i][j])
            tmp2.append(true_y[i][j])
    y_pred, y_true = np.array(tmp1), np.array(tmp2)
    acc = precision_score(y_true, y_pred, average='micro')
    p = precision_score(y_true, y_pred, average=average)
    r = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    return acc, p, r, f1

def prf_2nd_step(pair_id_all, pair_id, pred_y, fold = 0, save_dir = ''):
    pair_id_filtered = []
    for i in range(len(pair_id)):    #预测的
        if pred_y[i]:
            pair_id_filtered.append(pair_id[i])  #过滤掉
    def write_log():
        pair_to_y = dict(zip(pair_id, pred_y))
        g = open(save_dir+'pair_log_fold{}.txt'.format(fold), 'w')
        doc_id_b, doc_id_e = pair_id_all[0]/10000, pair_id_all[-1]/10000
        idx_1, idx_2 = 0, 0
        for doc_id in range(doc_id_b, doc_id_e+1):
            true_pair, pred_pair, pair_y = [], [], []
            line = str(doc_id) + ' '
            while True:
                p_id = pair_id_all[idx_1]
                d, p1, p2 = p_id/10000, p_id%10000/100, p_id%100
                if d != doc_id: break
                true_pair.append((p1, p2))
                line += '({}, {}) '.format(p1,p2)
                idx_1 += 1
                if idx_1 == len(pair_id_all): break
            line += '|| '
            while True:
                p_id = pair_id[idx_2]
                d, p1, p2 = p_id/10000, p_id%10000/100, p_id%100
                if d != doc_id: break
                if pred_y[idx_2]:
                    pred_pair.append((p1, p2))
                pair_y.append(pred_y[idx_2])
                line += '({}, {}) {} '.format(p1, p2, pred_y[idx_2])
                idx_2 += 1
                if idx_2 == len(pair_id): break
            if len(true_pair)>1:
                line += 'multipair '
                if true_pair == pred_pair:
                    line += 'good '
            line += '\n'
            g.write(line)
    if fold:
        write_log()
    keep_rate = len(pair_id_filtered)/(len(pair_id)+1e-8)
    s1, s2, s3 = set(pair_id_all), set(pair_id), set(pair_id_filtered)
    o_acc_num = len(s1 & s2)   #本来的真正正确的
    acc_num = len(s1 & s3)     #预测的真正正确的
    o_p, o_r = o_acc_num/(len(s2)+1e-8), o_acc_num/(len(s1)+1e-8)
    p, r = acc_num/(len(s3)+1e-8), acc_num/(len(s1)+1e-8)
    f1, o_f1 = 2*p*r/(p+r+1e-8), 2*o_p*o_r/(o_p+o_r+1e-8)
    
    return p, r, f1, o_p, o_r, o_f1, keep_rate
    
