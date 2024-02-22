# 用于正则表达式
import re
# 用于构建网络结构和函数的torch工具包
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# torch中预定义的优化方法工具包
import torch.optim as optim
import time
# 用于随机生成数据
import random
import matplotlib.pyplot as plt

# 设备选择, 我们可以选择在cuda或者cpu上运行你的代码
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 起始标志
SOS_token = 0
# 结束标志
EOS_token = 1
# 最大句子长度不能超过10个 (包含标点)
MAX_LENGTH = 10
# 数据文件路径
data_path = 'eng-fra.txt'


# 工具函数
def normalizeString(s):
    """字符串规范化函数, 参数s代表传入的字符串"""
    s = s.lower().strip()
    # 在.!?前加一个空格  这里的\1表示第一个分组   正则中的\num
    s = re.sub(r"([.!?])", r" \1", s)
    # s = re.sub(r"([.!?])", r" ", s)
    # 使用正则表达式将字符串中 不是 大小写字母和正常标点的都替换成空格
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# my_getdata() 清洗文本构建字典思路分析
# 1 按行读文件 open().read().strip().split(\n) my_lines
# 2 按行清洗文本 构建语言对 my_pairs[] tmppair[] for line in my_lines for s in line.split('\t')
# 2-1格式 [['英文', '法文'], ['英文', '法文'], ['英文', '法文'], ['英文', '法文']....]
# 2-2调用清洗文本工具函数normalizeString(s)
# 3 遍历语言对 构建英语单词字典 法语单词字典 my_pairs->pair->pair[0].splt(' ') pair[1].split(' ')->word
# 3-1 english_word2index english_word_n french_word2index french_word_n
# 其中 english_word2index = {0: "SOS", 1: "EOS"}  english_word_n=2
# 3-2 english_index2word french_index2word
# 4 返回数据的7个结果
# english_word2index, english_index2word, english_word_n,
# french_word2index, french_index2word, french_word_n, my_pairs
def my_getdata():

    # 1 按行读文件 open().read().strip().split(\n) my_lines
    my_lines = open(data_path, mode='r', encoding='utf-8').read().strip().split('\n')
    print('len my_lines-->', len(my_lines))

    # 2 按行清洗文本 构建语言对 my_pairs[] tmppair[] for line in my_lines for s in line.split('\t')
    # 2-1格式 [['英文', '法文'], ['英文', '法文'], ['英文', '法文'], ['英文', '法文']....]
    my_pairs, tmppair = [], []

    # for line in my_lines:
    #     for s in line.split('\t'):
    #         # 2-2调用清洗文本工具函数normalizeString(s)
    #         tmppair.append(normalizeString(s))
    #
    #     my_pairs.append(tmppair)
    #     tmppair = []

    my_pairs = [ [ normalizeString(s) for s in line.split('\t')]   for line in my_lines]
    print('my_pairs-->', len(my_pairs))
    print('my_pairs前4句话-->', my_pairs[0:4])

    print('my_pairs[8000][0]英文-->', my_pairs[8000][0])
    print('my_pairs[8000][1]法文-->', my_pairs[8000][1])

    # 3 遍历语言对 构建英语单词字典 法语单词字典 my_pairs->pair->pair[0].splt(' ') pair[1].split(' ')->word
    # 3-1 english_word2index english_word_n french_word2index french_word_n
    english_word2index = {"SOS" : 0, "EOS" : 1}
    english_word_n=2

    french_word2index = {"SOS" : 0, "EOS" : 1}
    french_word_n = 2

    for pair in my_pairs:
        # 添加英文word2index字典
        for word in pair[0].split(' '):
            if word not in english_word2index:
                english_word2index[word] = english_word_n
                english_word_n += 1

        # 添加法文word2index字典
        for word in pair[1].split(' '):
            if word not in french_word2index:
                french_word2index[word] = french_word_n
                french_word_n += 1

    # 3-2 english_index2word french_index2word
    english_index2word = {v:k for k,v in english_word2index.items()}
    french_index2word = {v:k for k,v in french_word2index.items()}

    print('english_index2word-->', len(english_index2word)) # 2803
    print('french_index2word-->', len(french_index2word))   # 4345

    # 4 返回数据的7个结果
    return english_word2index, english_index2word, english_word_n, \
           french_word2index, french_index2word, french_word_n, my_pairs


# 做成全局函数 读数据到内存 英文字典 法文字典 语言对
english_word2index, english_index2word, english_word_n, \
french_word2index, french_index2word, french_word_n, my_pairs = my_getdata()


# 原始数据 -> 数据源MyPairsDataset --> 数据迭代器DataLoader
# 构造数据源 MyPairsDataset，把语料xy 文本数值化 再转成tensor_x tensor_y
# 1 __init__(self, my_pairs)函数 设置self.my_pairs 条目数self.sample_len
# 2 __len__(self)函数  获取样本条数
# 3 __getitem__(self, index)函数 获取第几条样本数据
    # 按索引 获取数据样本 x y
    # 样本x 文本数值化   word2id  x.append(EOS_token)
    # 样本y 文本数值化   word2id  y.append(EOS_token)
    # 返回tensor_x, tensor_y
class MyPairsDataset(Dataset):
    def __init__(self, my_pairs):
        self.my_pairs = my_pairs            # 语言对
        self.sample_len = len(my_pairs)     # 样本条目数

    def __len__(self):
        return self.sample_len

    def __getitem__(self, index):

        # 按索引 获取数据样本 x y
        x = self.my_pairs[index][0]
        y = self.my_pairs[index][1]

        # 样本x 文本数值化   word2id  x.append(EOS_token)
        x = [english_word2index[word] for word in  x.split(' ')]
        x.append(EOS_token)
        tensor_x = torch.tensor(x, dtype=torch.long, device=device)

        # 样本y 文本数值化   word2id  y.append(EOS_token)
        y = [french_word2index[word] for word in y.split(' ')]
        y.append(EOS_token)
        tensor_y = torch.tensor(y, dtype=torch.long, device=device)

        # 返回tensor_x, tensor_y
        return tensor_x, tensor_y


def dm01_test_MyPairsDataset():

    # 1 实例化数据源
    mypairsdataset = MyPairsDataset(my_pairs)
    print('mypairsdataset-->', mypairsdataset)

    # 2 实例化dataloader
    mydataloader = DataLoader(dataset=mypairsdataset, batch_size=1, shuffle=True)
    print('mydataloader-->', mydataloader)

    # 3 遍历数据
    for x, y in mydataloader:
        print('x-->', x.shape, x)
        print('y-->', y.shape, y)
        break


# EncoderRNN类 实现思路分析：
# 1 init函数 定义2个层 self.embedding self.gru (batch_first=True)
    # def __init__(self, input_size, hidden_size): # 2803 256
# 2 forward(input, hidden)函数，返回output, hidden
    # 数据经过词嵌入层 数据形状 [1,6] --> [1,6,256]
    # 数据经过gru层 形状变化 gru([1,6,256],[1,1,256]) --> [1,6,256] [1,1,256]
# 3 初始化隐藏层输入数据 inithidden()
    # 形状 torch.zeros(1, 1, self.hidden_size, device=device)
# 构建基于GRU的编码器
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size): # 2803 256
        super(EncoderRNN, self).__init__()

        self.input_size = input_size    # 词嵌入单词的个数 2803
        self.hidden_size = hidden_size  # 单词的特征

        # 定义层1 词嵌入层 nn.Embedding(2803, 256)  # 创建了一个词向量矩阵
        self.embedding =  nn.Embedding(self.input_size,  self.hidden_size)

        # 定义层gru
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

    def forward(self, input, hidden):
        # 数据经过词嵌入层 数据形状 [1,6] --> [1,6,256]
        input  = self.embedding(input)

        # 数据经过gru层 形状变化 gru([1,6,256],[1,1,256]) --> [1,6,256] [1,1,256]
        output, hidden  = self.gru(input, hidden)

        # 返回数据
        return output, hidden

    def inithidden(self):
        return torch.zeros(1, 1,  self.hidden_size, device=device)


def dm02_test_EncoderRNN():

    # 1 实例化数据源
    mypairsdataset = MyPairsDataset(my_pairs)
    print('mypairsdataset-->', mypairsdataset)

    # 2 实例化dataloader
    mydataloader = DataLoader(dataset=mypairsdataset, batch_size=1, shuffle=True)
    print('mydataloader-->', mydataloader)

    # 3 实例化编码器
    # def __init__(self, input_size, hidden_size):  # 2803 256
    myencoerrnn = EncoderRNN(2803, 256)
    myencoerrnn = myencoerrnn.to(device)
    print('myencoerrnn->', myencoerrnn)

    # 3 遍历数据
    for x, y in mydataloader:
        print('x-->', x.shape, x)
        print('y-->', y.shape, y)

        # 1 一次性的喂数据
        hidden = myencoerrnn.inithidden()  # [1,6] -->[1,6,256] [1,1,256]
        output, hidden = myencoerrnn(x, hidden)
        # print('output-->', output.shape, output)
        # print('hidden-->', hidden.shape, hidden)  # [1,6,256] ---> [6, 256]
        print('一次性的喂数据最后一个隐藏层output-->', output[0][-1])

        # 2 一个单词一个单词的送数据
        hidden = myencoerrnn.inithidden() # 重新初始化init
        for i in range(x.shape[1]):
            tmpx = x[0][i].view(1, -1)  # [1,1]
            output, hidden = myencoerrnn(tmpx, hidden)
        print('一个字符一个字符的送数据output', output.shape, output)
        break


# 构建基于GRU的解码器
# DecoderRNN 类 实现思路分析：
# 解码器的作用：提取事物特征 进行分类（所以比 编码器 多了 线性层 和 softmax层）
# 1 init函数 定义四个层 self.embedding self.gru self.out self.softmax=nn.LogSoftmax(dim=-1)
    # def __init__(self, output_size, hidden_size): # 4345 256
# 2 forward(input, hidden)函数，返回output, hidden
    # 数据经过词嵌入层 数据形状 [1,1] --> [1,1,256]
    # 数据经过relu()层 input = F.relu(input)
    # 数据经过gru层 形状变化 gru([1,1,256],[1,1,256]) --> [1,1,256] [1,1,256]
    # 数据结果out层 形状变化 [1,1,256]->[1,256]-->[1,4345]
    # 返回 解码器分类output[1,4345]，最后隐层张量hidden[1,1,256]
# 3 初始化隐藏层输入数据 inithidden()
    # 形状 torch.zeros(1, 1, self.hidden_size, device=device)
class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size ): # 4345 256
        super(DecoderRNN, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size

        # 定义法文 词嵌入层
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        # 定义gru层
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

        # 定义out层 希望能映射成4345种可能
        self.out = nn.Linear(self.hidden_size,  self.output_size)

        # 定义softmax
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):

        # 数据经过词嵌入层 数据形状 [1,1] --> [1,1,256]
        input = self.embedding(input)

        # 数据经过relu()层 input = F.relu(input)
        input = F.relu(input)

        # 数据经过gru层 形状变化 gru([1,1,256],[1,1,256]) --> [1,1,256] [1,1,256]
        output, hidden = self.gru(input, hidden)

        # 数据结果out层 形状变化 [1,1,256]->[1,256]-->[1,4345]
        output = self.out( output[0])
        output = self.softmax(output)

        # 返回 解码器分类output[1,4345]，最后隐层张量hidden[1,1,256]
        return output, hidden

    def inithidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def dm03_test_DecoderRNN():

    # 1 实例化数据源
    mypairsdataset = MyPairsDataset(my_pairs)
    print('mypairsdataset-->', mypairsdataset)

    # 2 实例化dataloader
    mydataloader = DataLoader(dataset=mypairsdataset, batch_size=1, shuffle=True)
    print('mydataloader-->', mydataloader)

    # 3-1 实例化编码器
    myencoerrnn = EncoderRNN(2803, 256)
    myencoerrnn = myencoerrnn.to(device)
    print('myencoerrnn->', myencoerrnn)

    # 3-2 实例化解码
    mydecoderrnn = DecoderRNN(4345, 256)
    mydecoderrnn = mydecoderrnn.to(device)
    print('mydecoderrnn->', mydecoderrnn)

    # 4 遍历数据
    for x, y in mydataloader:
        print('x-->', x.shape, x)
        print('y-->', y.shape, y)

        # 1 先对英文进行编码 一次性的喂数据
        hidden = myencoerrnn.inithidden()  # [1,6] -->[1,6,256] [1,1,256]
        output, hidden = myencoerrnn(x, hidden)
        print('中间语义张量C output_c-->', output.shape)
        # print('hidden-->', hidden.shape, hidden)  # [1,6,256] ---> [6, 256]
        # print('一次性的喂数据最后一个隐藏层output-->', output[0][-1])

        # 2 在对法文进行解码，一个单词一个单词的送数据
        # hidden = mydecoderrnn.inithidden()  # 重新初始化init 不需要！
        for i in range(y.shape[1]):
            tmpy = y[0][i].view(1, -1)  # [1,1]
            output, hidden = mydecoderrnn(tmpy, hidden)
            print('一个字符一个字符解码 每个时间步有4345种可能', output.shape)
        break


# 相对传统RNN解码 AttnDecoderRNN类多了注意力机制,需要构建QKV
# 1 在init函数中 (self, output_size, hidden_size, dropout_p=0.1, max_length=MAX_LENGTH)
    # 增加层 self.attn  self.attn_combine  self.dropout=nn.Dropout(self.dropout_p)
# 2 增加函数 attentionQKV(self, Q, K, V)
# 3 函数forward(self, input, hidden, encoder_outputs)
    # encoder_outputs 每个时间步解码准备qkv 调用attentionQKV
    # 函数返回值 output, hidden, attn_weights
# 4 调用需要准备中间语义张量C encoder_output_c
class AttnDecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, dropout_p=0.1, max_length=MAX_LENGTH): # 4345 256
        super(AttnDecoderRNN, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p          # add
        self.max_length = max_length        # add

        # 定义法文 词嵌入层
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        # 定义gru层
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

        # 定义out层 希望能映射成4345种可能
        self.out = nn.Linear(self.hidden_size,  self.output_size)

        # 定义softmax
        self.softmax = nn.LogSoftmax(dim=-1)

        # 增加注意力机制相关线性层
        # 线性层1 注意力权重分布层
        self.attn = nn.Linear(self.hidden_size + self.hidden_size, self.max_length)     # add

        # 线性层2：注意力结果表示按照指定维度进行输出层
        self.attn_combine = nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size) # add

        self.dropout = nn.Dropout(self.dropout_p)       # add

    # 添加注意力机制 专有函数
    def attentionQKV(self, Q, K, V):   # add

        # 1 求查询张量q的注意力权重分布, attn_weights[1,32]
        # tmp1 = torch.cat((Q[0], K[0]), dim=-1)  # [1,1,32], [1,1,32] ->[1,32],[1,32] --> [1,64]
        # tmp2 = self.attn(tmp1)  # [1,64] --> [1,32]
        # tmp3 = torch.softmax(tmp2, dim=-1) # [1,32] 数据归一化 权重分布
        attn_weights = torch.softmax(self.attn( torch.cat((Q[0], K[0]), dim=-1)), dim=-1)

        # 2 求查询张量q的注意力结果表示 bmm运算, attn_applied[1,1,64]
        # [1,32]-->  [1,1,32] @ [1,32,64] --> [1,1,64]
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), V)

        # 3 q 与 attn_applied 融合，再按照指定维度输出 output[1,1,32]
        # 3-1 q 与 attn_applied 融合  [1,1,32],[1,1,64] --> [1,32] [1,64] ==> [1,96]
        output = torch.cat((Q[0], attn_applied[0]), dim=-1)

        # 3-2 再按照指定维度输出 [1,96] --> [1,32] --> [1,1,32]
        output = self.attn_combine(output).unsqueeze(0)

        # 返回注意力结果表示output:[1,1,32], 注意力权重分布attn_weights:[1,32]
        return output, attn_weights

    def forward(self, input, hidden, encoder_outputs):

        # 数据经过词嵌入层 数据形状 [1,1] --> [1,1,256]
        input = self.embedding(input)
        input = self.dropout(input)

        # 让数据进行qkv运算
        input, attn_weights = self.attentionQKV(input, hidden, encoder_outputs.unsqueeze(0))

        # 数据经过relu()层 input = F.relu(input)
        input = F.relu(input)

        # 数据经过gru层 形状变化 gru([1,1,256],[1,1,256]) --> [1,1,256] [1,1,256]
        output, hidden = self.gru(input, hidden)

        # 数据结果out层 形状变化 [1,1,256]->[1,256]-->[1,4345]
        output = self.out( output[0])
        output = self.softmax(output)

        # 返回 解码器分类output[1,4345]，最后隐层张量hidden[1,1,256]
        return output, hidden, attn_weights

    def inithidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# encode_output_c = torch.zeros(10, 256, device=device)
def dm04_test_AttnDecoderRNN():
    # 1 实例化数据源
    mypairsdataset = MyPairsDataset(my_pairs)
    print('mypairsdataset-->', mypairsdataset)

    # 2 实例化dataloader
    mydataloader = DataLoader(dataset=mypairsdataset, batch_size=1, shuffle=True)
    print('mydataloader-->', mydataloader)

    # 3-1 实例化编码器
    myencoerrnn = EncoderRNN(2803, 256)
    myencoerrnn = myencoerrnn.to(device)
    print('myencoerrnn->', myencoerrnn)

    # 3-2 实例化注意力机制的解码器
    myattndecoderrnn = AttnDecoderRNN(output_size=4345, hidden_size=256, dropout_p=0.1, max_length=10)
    myattndecoderrnn = myattndecoderrnn.to(device)
    print('myattndecoderrnn->', myattndecoderrnn)

    # 4 遍历数据
    for x, y in mydataloader:
        print('x-->', x.shape, x)
        print('y-->', y.shape, y)

        # 1 先对英文进行编码 一次性的喂数据
        hidden = myencoerrnn.inithidden()  # [1,6] -->[1,6,256] [1,1,256]
        output, hidden = myencoerrnn(x, hidden)
        print('中间语义张量C output_c-->', output.shape)
        # print('hidden-->', hidden.shape, hidden)  # [1,6,256] ---> [6, 256]
        # print('一次性的喂数据最后一个隐藏层output-->', output[0][-1])

        # 固定化一个中间语义张量C
        encode_output_c = torch.zeros(10, 256, device=device)
        for idx in range(output.shape[1]):
            encode_output_c[idx] = output[0, idx]

        # 2 在对法文进行解码，一个单词一个单词的送数据
        # hidden = mydecoderrnn.inithidden()  # 重新初始化init 不需要！
        for i in range(y.shape[1]):
            tmpy = y[0][i].view(1, -1)  # [1,1]
            output, hidden, attn_weights = myattndecoderrnn(tmpy, hidden, encode_output_c)
            print('一个字符一个字符解码 每个时间步有4345种可能', output.shape)
            print('attn_weights.shape', attn_weights.shape)
        break


# 全局变量
epochs = 1                      # 训练轮次
mylr = 1e-4                     # 学习率
teacher_forcing_ratio = 0.5     # 策略
print_interval_num = 100        # 每100次迭代 打印日志间隔变量
plot_interval_num = 100         # 每100次迭代 画图间隔变量


# 内部迭代训练函数Train_Iters
# 1 编码 encode_output, encode_hidden = my_encoderrnn(x, encode_hidden)
# 数据形状 eg [1,6],[1,1,256] --> [1,6,256],[1,1,256]

# 2 解码参数准备和解码
# 解码参数1 固定长度C encoder_outputs_c = torch.zeros(MAX_LENGTH, my_encoderrnn.hidden_size, device=device)
# 解码参数2 decode_hidden # 解码参数3 input_y = torch.tensor([[SOS_token]], device=device)
# 数据形状 [1,1],[1,1,256],[10,256] ---> [1,4345],[1,1,256],[1,10]
# output_y, decode_hidden, attn_weight = my_attndecoderrnn(input_y, decode_hidden, encode_output_c)
# 计算损失 target_y = y[0][idx].view(1)
# 每个时间步处理 for idx in range(y_len): 处理三者之间关系input_y output_y target_y

# 3 训练策略 use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
# teacher_forcing  把样本真实值y作为下一次输入 input_y = y[0][idx].view(1, -1)
# not teacher_forcing 把预测值y作为下一次输入
# topv,topi = output_y.topk(1) # if topi.squeeze().item() == EOS_token: break input_y = topi.detach()

# 4 其他 # 计算损失  # 梯度清零 # 反向传播  # 梯度更新 # 返回 损失列表myloss.item()/y_len
def Train_Iters(x, y, my_encoderrnn, my_attndecoderrnn, myadam_encode, myadam_decode, mycrossentropyloss):

    # 1 编码 encode_output, encode_hidden = my_encoderrnn(x, encode_hidden)
    encode_hidden =  my_encoderrnn.inithidden()
    # 数据形状 eg [1,6],[1,1,256] --> [1,6,256],[1,1,256]
    encode_output, encode_hidden =  my_encoderrnn(x, encode_hidden)

    # 2 解码参数准备和解码 q k v
    # 解码参数1 固定长度C encoder_outputs_c = torch.zeros(MAX_LENGTH, my_encoderrnn.hidden_size, device=device)
    encoder_output_c = torch.zeros(MAX_LENGTH, my_encoderrnn.hidden_size, device=device)
    for idx in range(x.shape[1]):
        encoder_output_c[idx] = encode_output[0, idx]

    # 解码参数2
    decode_hidden = encode_hidden

    # 解码参数3 input_y = torch.tensor([[SOS_token]], device=device)
    input_y = torch.tensor([[SOS_token]], device=device )

    y_len = y.shape[1]
    myloss = 0
    for idx in range(y_len):
        # 数据形状 [1,1],[1,1,256],[10,256] ---> [1,4345],[1,1,256],[1,10]
        output_y, decode_hidden, attn_weight =   my_attndecoderrnn(input_y, decode_hidden, encoder_output_c)
        target_y = y[0][idx].view(1)
        myloss = myloss + mycrossentropyloss(output_y, target_y)
        input_y = y[0][idx].view(1, -1)

    # 梯度清零
    myadam_encode.zero_grad()
    myadam_decode.zero_grad()

    # 反向传播
    myloss.backward()

    # 梯度更新
    myadam_encode.step()
    myadam_decode.step()

    # 返回 损失列表myloss.item()/y_len
    return myloss.item() / y_len


# Train_seq2seq() 思路分析
# 实例化 mypairsdataset对象  实例化 mydataloader
# 实例化编码器 my_encoderrnn 实例化解码器 my_attndecoderrnn
# 实例化编码器优化器 myadam_encode 实例化解码器优化器 myadam_decode
# 实例化损失函数 mycrossentropyloss = nn.NLLLoss()
# 定义模型训练的参数
# epoches mylr=1e4 teacher_forcing_ratio print_interval_num  plot_interval_num (全局)
# plot_loss_list = [] (返回) print_loss_total plot_loss_total starttime (每轮内部)

# 外层for循环 控制轮数 for epoch_idx in range(1, 1+epochs)
# 内层for循环 控制迭代次数 # for item, (x, y) in enumerate(mydataloader, start=1)
#   调用内部训练函数 Train_Iters(x, y, my_encoderrnn, my_attndecoderrnn, myadam_encode, myadam_decode, mycrossentropyloss)
# 计算辅助信息
#   计算打印屏幕间隔损失-每隔1000次 # 计算画图间隔损失-每隔100次
#   每个轮次保存模型 torch.save(my_encoderrnn.state_dict(), PATH1)
#   所有轮次训练完毕 画损失图 plt.figure() .plot(plot_loss_list) .save('x.png') .show()
def Train_seq2seq():

    # 实例化 mypairsdataset对象  实例化 mydataloader
    mypairsdataset = MyPairsDataset(my_pairs)
    mydataloader = DataLoader(dataset=mypairsdataset, batch_size=1, shuffle=True)

    # 实例化编码器 my_encoderrnn 实例化解码器 my_attndecoderrnn
    my_encoderrnn = EncoderRNN(2803, 256)
    my_encoderrnn = my_encoderrnn.to(device=device)
    my_attndecoderrnn = AttnDecoderRNN(output_size=4345, hidden_size=256, dropout_p=0.1, max_length=10)
    my_attndecoderrnn = my_attndecoderrnn.to(device=device)

    # 实例化编码器优化器 myadam_encode 实例化解码器优化器 myadam_decode
    myadam_encode = optim.Adam(my_encoderrnn.parameters(), lr=mylr)
    myadam_decode = optim.Adam(my_attndecoderrnn.parameters(), lr=mylr)

    # 实例化损失函数 mycrossentropyloss = nn.NLLLoss()
    mycrossentropyloss = nn.NLLLoss()

    # 定义模型训练的参数
    plot_loss_list = []
    # (返回) print_loss_total plot_loss_total starttime (每轮内部)

    # 外层for循环 控制轮数 for epoch_idx in range(1, 1+epochs)
    for epoch_idx in range(1, 1+epochs):

        print_loss_total, plot_loss_total = 0.0, 0.0
        starttime = time.time()

        # 内层for循环 控制迭代次数 # for item, (x, y) in enumerate(mydataloader, start=1)
        for item ,(x, y) in enumerate(mydataloader, start=1):
            # 调用内部训练函数
            myloss = Train_Iters(x, y, my_encoderrnn, my_attndecoderrnn, myadam_encode, myadam_decode, mycrossentropyloss)
            print_loss_total += myloss
            plot_loss_total += myloss

            # 计算辅助信息
            # 计算打印屏幕间隔损失-每隔100次
            if item % print_interval_num == 0:
                print_loss_avg = print_loss_total / print_interval_num
                # 将总损失归0
                print_loss_total = 0
                # 打印日志，日志内容分别是：训练耗时，当前迭代步，当前进度百分比，当前平均损失
                print('轮次%d  损失%.6f 时间:%d' % (epoch_idx, print_loss_avg, time.time() - starttime))

            # 计算画图间隔损失-每隔100次
            if item % plot_interval_num == 0:
                # 通过总损失除以间隔得到平均损失
                plot_loss_avg = plot_loss_total / plot_interval_num
                # 将平均损失添加plot_loss_list列表中
                plot_loss_list.append(plot_loss_avg)
                # 总损失归0
                plot_loss_total = 0

        # 每个轮次保存模型
        torch.save(my_encoderrnn.state_dict(), 'my_encoderrnn_%d.pth' % epoch_idx)
        torch.save(my_attndecoderrnn.state_dict(), 'my_attndecoderrnn_%d.pth' % epoch_idx)

    # 所有轮次训练完毕 画损失图
    plt.figure()
    plt.plot(plot_loss_list)
    plt.savefig('seq2seq_lost.png')
    plt.show()


if __name__ == '__main__':
    # dm01_test_MyPairsDataset()
    # dm02_test_EncoderRNN()
    # dm03_test_DecoderRNN()
    # dm04_test_AttnDecoderRNN()
    Train_seq2seq()
    print('Seq2seq End')
