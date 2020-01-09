a='(7,8)'
pairs = eval('[' + a.strip() + ']')
print(a,pairs,type(a),type(pairs))

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
aa=0
bb=0

while True:

    line=inputFile.readline()
    if line == '': break
    line = line.strip().split()
    # print(type(line[-1]))
    a=line[-1]
    # if a in dict:
    #     dict[a]+=1
    # else:
    #     dict[a]=1
    # inputFile.readline()
    # for i in range(eval(a)):
    #     inputFile.readline()
    pairs = eval('[' + inputFile.readline().strip() + ']')  # 执行字符串表达式
    for i in pairs:
        aa+=1

    if eval(a)>30:
        bb+=900
    else:
        bb+=eval(a)*eval(a)

    for i in range(eval(a)):
        inputFile.readline()

print(aa,bb)