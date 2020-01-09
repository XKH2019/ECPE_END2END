inputFile = open('data_combine/fold3_train.txt', 'r',encoding="UTF-8")
# line = inputFile.readline()
# line = inputFile.readline()
# line = inputFile.readline().strip().split()
# line = inputFile.readline().strip().split(',')

# print(line)
# words=line[-1]
# # print(words.type())
# print(words)

# words=words.split()
# print(words)
dict={}


while True:

    line=inputFile.readline()
    if line == '': break
    line = line.strip().split()
    print(type(line[-1]))
    a=line[-1]
    if a in dict:
        dict[a]+=1
    else:
        dict[a]=1
    inputFile.readline()
    for i in range(eval(a)):
        inputFile.readline()

print(dict)
list1= sorted(dict.items(),key=lambda x:x[1])
print(list1)
list2= sorted(dict.items(),key=lambda x:x[1],reverse=True)
print(list2)

list3= sorted(dict.items(),key=lambda x:x[0])
print(list3)


l1=dict.keys()
l2=dict.values()
l3=dict.items()
l4=sorted(l3)
print(l4)



import numpy as np
m=np.zeros((10, 5),dtype=np.int32)
n=m[1]
print(n)
for i in range(2,7 ):
    print(i)

a=[]
a.append(m)
print(a,type(a))
a=np.array(a)
print(a,type(a))

# for i in range(10):

a=np.arange(25).reshape(5,5)
blue=a[1::2,0:3:2]
print(blue)

for i in range(1,2):
    print('nihao',i)
list1 = ['Google', 'Runoob', 'Taobao']
list2=list(range(5)) # 创建 0-4 的列表
list1.extend(list2)  # 扩展列表
print ("扩展后的列表：", list1)
list1.append([1])  # 扩展列表
print ("扩展后的列表：", list1)


a='155'
a=int (a)
print(a,type(a))

while True:
    line = inputFile.readline()
    if line == '': break
    line = line.strip().split()
    # doc_id.append(line[0])  # 文档id
    doc_id = int(line[0])
    d_len = int(line[1])  # 文档长度
    pairs = eval('[' + inputFile.readline().strip() + ']')  # 执行字符串表达式
    pair_id_all.extend([doc_id * 10000 + p[0] * 100 + p[1] for p in pairs])  # 添加每个文档的预测的pair的权重化值

    doc_len.append(d_len)  # 第一个文档子句个数
    y_pairs.append(pairs)  # 第一个文档的子句对
    # pos, cause = zip(*pairs)  # 第一个文档的情感句和原因句  反zip 返回一个列表
    # y_po, y_ca, sen_len_tmp, x_tmp = np.zeros((max_doc_len, 2)), np.zeros((max_doc_len, 2)), np.zeros(max_doc_len,
    #                                                                                                   dtype=np.int32), np.zeros(
    #     (max_doc_len, max_sen_len), dtype=np.int32)

    sen_len_tmp, x_tmp = np.zeros(max_doc_len, dtype=np.int32), np.zeros((max_doc_len, max_sen_len), dtype=np.int32)

    for i in range(d_len):
        # y_po[i][int(i + 1 in pos)] = 1
        # y_ca[i][int(i + 1 in cause)] = 1
        if i >= max_doc_len:
            inputFile.readline()
        else:
            words = inputFile.readline().strip().split(',')[-1]
            sen_len_tmp[i] = min(len(words.split()), max_sen_len)  # zijiu的真实长度  75长数组

            for j, word in enumerate(words.split()):
                if j >= max_sen_len:
                    n_cut += 1  # 单词数量
                    break

                x_tmp[i][j] = int(word_idx[word])  # 将文档将子句装换成索引

    # y_position.append(y_po)
    # y_cause.append(y_ca)
    x.append(x_tmp)
    sen_len.append(sen_len_tmp)

    while True:
        line = inputFile.readline()  # diyihang
        if line == '': break
        line = line.strip().split()
        doc_id = int(line[0])
        d_len = int(line[1])

        pairs = eval(inputFile.readline().strip())  # yuce的pair列表  dierhang

        pair_id_all.extend([doc_id * 10000 + p[0] * 100 + p[1] for p in pairs])  # 添加每个文档的预测的pair的权重化值

        sen_len_tmp, x_tmp = np.zeros(max_doc_len, dtype=np.int32), np.zeros((max_doc_len, max_sen_len),
                                                                             dtype=np.int32)  # （75） ，（75，30）

        pos_list, cause_list = [], []
        for i in range(d_len):
            line = inputFile.readline().strip().split(',')
            if int(line[1].strip()) > 0:
                pos_list.append(i + 1)
            if int(line[2].strip()) > 0:
                cause_list.append(i + 1)
            words = line[-1]
            sen_len_tmp[i] = min(len(words.split()), max_sen_len)  # 每个句子的实际长度
            for j, word in enumerate(words.split()):
                if j >= max_sen_len:
                    n_cut += 1  # 切分句子的数目
                    break
                x_tmp[i][j] = int(word_idx[word])  # 存储句子
        # 结果 ： 真实的pair标签 预测的情感 原因标签 每个句子的实际长度 每个句子的index

        for i in pos_list:  # 加过1
            for j in cause_list:
                pair_id_cur = doc_id * 10000 + i * 100 + j
                pair_id.append(pair_id_cur)  # 预测 的pair值

                y.append([0, 1] if pair_id_cur in pair_id_all else [1, 0])  # 如果预测正确 则添加【0，1】 y中包括所有预测的pair    创造的真是标签

                x.append([x_tmp[i - 1], x_tmp[j - 1]])

                sen_len.append([sen_len_tmp[i - 1], sen_len_tmp[j - 1]])
                distance.append(j - i + 100)  # 加100？