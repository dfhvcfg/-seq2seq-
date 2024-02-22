import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
import torch.optim as optim
import time


# 设备选择, 我们可以选择在cuda或者cpu上运行你的代码
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 起始标志
SOS_token = 0
# 结束标志
EOS_token = 1
# 最大句子长度不能超过10个 (包含标点)
MAX_LENGTH = 10
# 数据文件路径
data_path = './data/eng-fra-v2.txt'

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

def my_getdata():

    # 1 按行读文件 open().read().strip().split(\n)
    my_lines = open(data_path, encoding='utf-8').read().strip().split('\n')
    print('my_lines--->', len(my_lines))

    # 2 按行清洗文本 构建语言对pairs
    # 格式 [['英文', '法文'], ['英文', '法文'], ['英文', '法文'], ['英文', '法文']....]
    # tmp_pair = []
    # my_pairs = []
    # for l in lines:
    #     for s in l.split('\t'):
    #         tmp_pair.append(normalizeString(s))
    #     my_pairs.append(tmp_pair)
    #     tmp_pair = []
    my_pairs = [[normalizeString(s) for s in l.split('\t')] for l in my_lines]
    print('len(pairs)--->', len(my_pairs))

    # 打印前4条数据
    print(my_pairs[:4])

    # 打印第8000条的英文 法文数据
    print('my_pairs[8000][0]--->', my_pairs[8000][0])
    print('my_pairs[8000][1]--->', my_pairs[8000][1])

    # 3 遍历语言对 构建英语单词字典 法语单词字典
    # 3-1 english_word2index english_word_n french_word2index french_word_n
    english_word2index = {"SOS":0, "EOS":1}
    english_word_n = 2

    french_word2index = {"SOS":0, "EOS":1}
    french_word_n = 2

    # 遍历语言对 获取英语单词字典 法语单词字典
    for pair in my_pairs:
       for word in pair[0].split(' '):
           if word not in english_word2index:
               english_word2index[word] = english_word_n
               english_word_n += 1

       for word in pair[1].split(' '):
           if word not in french_word2index:
               french_word2index[word] = french_word_n
               french_word_n += 1

    # 3-2 english_index2word french_index2word
    english_index2word = {v:k for k, v in english_word2index.items()}
    french_index2word = {v:k for k, v in french_word2index.items()}

    print('len(english_word2index)-->', len(english_word2index))
    print('len(french_word2index)-->', len(french_word2index))
    print('english_word_n--->', english_word_n, 'french_word_n-->', french_word_n)

    return english_word2index, english_index2word, english_word_n, french_word2index, french_index2word, french_word_n, my_pairs

# 全局函数 获取英语单词字典 法语单词字典 语言对列表my_pairs
english_word2index, english_index2word, english_word_n, \
    french_word2index, french_index2word, french_word_n, \
    my_pairs = my_getdata()

# 原始数据 -> 数据源MyPairsDataset --> 数据迭代器DataLoader
# 构造数据源 MyPairsDataset，把语料转换成x y
# 1 init函数 设置self.my_pairs 条目数self.sample_len
# 2 __len__(self)函数  获取样本条数
# 3 __getitem__(self, index)函数 获取第几条样本数据
#       按索引 获取数据样本 x y
#       样本x 文本数值化   word2id  x.append(EOS_token)
#       样本y 文本数值化   word2id  y.append(EOS_token)
#       返回tensor_x, tensor_y

class MyPairsDataset(Dataset):
    def __init__(self, my_pairs):
        # 样本x
        self.my_pairs = my_pairs

        # 样本条目数
        self.sample_len = len(my_pairs)

    # 获取样本条数
    def __len__(self):
        return self.sample_len

    # 获取第几条 样本数据
    def __getitem__(self, index):

        # 对index异常值进行修正 [0, self.sample_len-1]
        index = min(max(index, 0), self.sample_len-1)

        # 按索引获取 数据样本 x y
        x = self.my_pairs[index][0]
        y = self.my_pairs[index][1]

        # 样本x 文本数值化
        x = [english_word2index[word] for word in x.split(' ')]
        x.append(EOS_token)
        tensor_x = torch.tensor(x, dtype=torch.long, device=device)

        # 样本y 文本数值化
        y = [french_word2index[word] for word in y.split(' ')]
        y.append(EOS_token)
        tensor_y = torch.tensor(y, dtype=torch.long, device=device)
        # 注意 tensor_x tensor_y都是一维数组，通过DataLoader拿出数据是二维数据
        # print('tensor_y.shape===>', tensor_y.shape, tensor_y)

        # 返回结果
        return tensor_x, tensor_y



# EncoderRNN类 实现思路分析：
# 1 init函数 准备三个层 self.rnn self.embedding self.gru (batch_first=True)
#    def __init__(self, input_size, hidden_size): # 2803 256

# 2 forward(input, hidden)函数，返回output, hidden
#   数据经过词嵌入层 数据形状 [1,6] --> [1,6,256]
#   数据经过gru层 形状变化 gru([1,6,256],[1,1,256]) --> [1,6,256] [1,1,256]

# 3 初始化隐藏层输入数据 inithidden()
#   形状 torch.zeros(1, 1, self.hidden_size, device=device)

# 构建基于GRU的编码器
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):

        # input_size 编码器 词嵌入层单词数 eg：2803
        # hidden_size 编码器 词嵌入层每个单词的特征数 eg 256
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 实例化nn.Embedding层
        self.embedding = nn.Embedding(input_size, hidden_size)

        # 实例化nn.GRU层 注意参数batch_first=True
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input, hidden):

        # 数据经过词嵌入层 数据形状 [1,6] --> [1,6,256]
        output = self.embedding(input)

        # 数据经过gru层 数据形状 gru([1,6,256],[1,1,256]) --> [1,6,256] [1,1,256]
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def inithidden(self):
        # 将隐层张量初始化成为1x1xself.hidden_size大小的张量
        return torch.zeros(1, 1, self.hidden_size, device=device)



# 构建基于GRU的解码器
# DecoderRNN 类 实现思路分析：
# 1 init函数 定义四个层 self.embedding self.gru self.out self.softmax=nn.LogSoftmax(dim=-1)
#    def __init__(self, output_size, hidden_size): # 4345 256

# 2 forward(input, hidden)函数，返回output, hidden
#   数据经过词嵌入层 数据形状 [1,1] --> [1,1,256]
#   数据经过relu()层 output = F.relu(output)
#   数据经过gru层 形状变化 gru([1,1,256],[1,1,256]) --> [1,1,256] [1,1,256]
#   返回 解码器分类output[1,4345]，最后隐层张量hidden[1,1,256]

# 3 初始化隐藏层输入数据 inithidden()
#   形状 torch.zeros(1, 1, self.hidden_size, device=device)
class DecoderRNN(nn.Module):

    def __init__(self, output_size, hidden_size):

        # output_size 编码器 词嵌入层单词数 eg：4345
        # hidden_size 编码器 词嵌入层每个单词的特征数 eg 256
        super(DecoderRNN, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size

        # 实例化词嵌入层
        self.embedding = nn.Embedding(output_size, hidden_size)

        # 实例化gru层，输入尺寸256 输出尺寸256
        # 因解码器一个字符一个字符的解码 batch_first=True 意义不大
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

        # 实例化线性输出层out 输入尺寸256 输出尺寸4345
        self.out = nn.Linear(hidden_size, output_size)

        # 实例化softomax层 数值归一化 以便分类
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):

        # 数据经过词嵌入层
        # 数据形状 [1,1] --> [1,1,256] or [1,6]--->[1,6,256]
        output = self.embedding(input)

        # 数据结果relu层使Embedding矩阵更稀疏，以防止过拟合
        output = F.relu(output)

        # 数据经过gru层
        # 数据形状 gru([1,1,256],[1,1,256]) --> [1,1,256] [1,1,256]
        output, hidden = self.gru(output, hidden)

        # 数据经过softmax层 归一化
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def inithidden(self):

        # 将隐层张量初始化成为1x1xself.hidden_size大小的张量
        return torch.zeros(1, 1, self.hidden_size, device=device)


# 构建基于GRU和Attention的解码器
# AttnDecoderRNN 类 实现思路分析：
# 1 init函数 定义六个层
#   self.embedding self.attn  self.attn_combine
#   self.gru self.out self.softmax=nn.LogSoftmax(dim=-1)
#   def __init__(self, output_size, hidden_size, dropout_p=0.1, max_length=MAX_LENGTH):: # 4345 256

# 2 forward(input, hidden)函数，返回output, hidden
#   数据经过词嵌入层 数据形状 [1,1] --> [1,1,256]
#   1 求查询张量q的注意力权重分布, attn_weights[1,10]
#   2 求查询张量q的注意力结果表示 bmm运算, attn_applied[1,1,256]
#   3 q 与 attn_applied 融合，经过层attn_combine 按照指定维度输出 output[1,1,256]
#   数据经过relu()层 output = F.relu(output)
#   数据经过gru层 形状变化 gru([1,1,256],[1,1,256]) --> [1,1,256] [1,1,256]
#   返回 # 返回解码器分类output[1,4345]，最后隐层张量hidden[1,1,256] 注意力权重张量attn_weights[1,10]

# 3 初始化隐藏层输入数据 inithidden()
#   形状 torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, dropout_p=0.1, max_length=MAX_LENGTH):

        # output_size   编码器 词嵌入层单词数 eg：4345
        # hidden_size   编码器 词嵌入层每个单词的特征数 eg 256
        # dropout_p     置零比率，默认0.1,
        # max_length    最大长度10
        super(AttnDecoderRNN, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # 定义nn.Embedding层 nn.Embedding(4345,256)
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        # 定义线性层1：求q的注意力权重分布
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)

        # 定义线性层2：q+注意力结果表示融合后，在按照指定维度输出
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # 定义dropout层
        self.dropout = nn.Dropout(self.dropout_p)

        # 定义gru层
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

        # 定义out层 解码器按照类别进行输出(256,4345)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def attentionQKV(self, q, k, v):

        # 全是三维张量
        # 1 计算q的权重分布
        # tmpq = torch.cat((q[0], k[0]), dim=-1)
        # tmpq = self.attn(tmpq)
        # tmpq = F.softmax(tmpq, dim=-1)
        # print('tmpq-->', tmpq)
        attn_weights = F.softmax(self.attn(torch.cat((q[0], k[0]), dim=-1)), dim=-1)
        print('attn_weights--->', attn_weights.shape)

        # 2 bmm运算注意力结果表示
        tmpq = torch.bmm(attn_weights.unsqueeze(0), v.unsqueeze(0))

        # 3 q与attn_applied 进行融合
        tmpq = torch.cat(q[0], tmpq[0], dim=-1)
        tmpq =  self.attn_combine(tmpq)
        return tmpq, attn_weights

    def forward(self, input, hidden, encoder_outputs):
        # input代表q [1,1] 二维数据 hidden代表k [1,1,256] encoder_outputs代表v [10,256]

        # 数据经过词嵌入层
        # 数据形状 [1,1] --> [1,1,256]
        embedded = self.embedding(input)

        # 使用dropout进行随机丢弃，防止过拟合
        embedded = self.dropout(embedded)


        # 1 求查询张量q的注意力权重分布, attn_weights[1,10]
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        # 2 求查询张量q的注意力结果表示 bmm运算, attn_applied[1,1,256]
        # [1,1,10],[1,10,256] ---> [1,1,256]
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        # 3 q 与 attn_applied 融合，再按照指定维度输出 output[1,1,256]
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        # 查询张量q的注意力结果表示 使用relu激活
        output = F.relu(output)

        # 查询张量经过gru、softmax进行分类结果输出
        # 数据形状[1,1,256],[1,1,256] --> [1,1,256], [1,1,256]
        output, hidden = self.gru(output, hidden)
        # 数据形状[1,1,256]->[1,256]->[1,4345]
        output = F.log_softmax(self.out(output[0]), dim=1)

        # 返回解码器分类output[1,4345]，最后隐层张量hidden[1,1,256] 注意力权重张量attn_weights[1,10]
        return output, hidden, attn_weights

    def inithidden(self):
        # 将隐层张量初始化成为1x1xself.hidden_size大小的张量
        return torch.zeros(1, 1, self.hidden_size, device=device)


# 模型评估代码与模型预测代码类似，需要注意使用with torch.no_grad()
# 模型预测时，第一个时间步使用SOS_token作为输入 后续时间步采用预测值作为输入，也就是自回归机制
def Seq2Seq_Evaluate(x, my_encoderrnn, my_attndecoderrnn):
    with torch.no_grad():
        # 1 编码：一次性的送数据
        encode_hidden = my_encoderrnn.inithidden()
        encode_output, encode_hidden = my_encoderrnn(x, encode_hidden)

        # 2 解码参数准备
        # 解码参数1 固定长度中间语义张量c
        encoder_outputs_c = torch.zeros(MAX_LENGTH, my_encoderrnn.hidden_size, device=device)
        x_len = x.shape[1]
        for idx in range(x_len):
            encoder_outputs_c[idx] = encode_output[0, idx]

        # 解码参数2 最后1个隐藏层的输出 作为 解码器的第1个时间步隐藏层输入
        decode_hidden = encode_hidden

        # 解码参数3 解码器第一个时间步起始符
        input_y = torch.tensor([[SOS_token]], device=device)

        # 3 自回归方式解码
        # 初始化预测的词汇列表
        decoded_words = []
        # 初始化attention张量
        decoder_attentions = torch.zeros(MAX_LENGTH, MAX_LENGTH)
        for idx in range(MAX_LENGTH): # note:MAX_LENGTH=10
            output_y, decode_hidden, attn_weights = my_attndecoderrnn(input_y, decode_hidden, encoder_outputs_c)
            # 预测值作为为下一次时间步的输入值
            topv, topi = output_y.topk(1)
            decoder_attentions[idx] = attn_weights

            # 如果输出值是终止符，则循环停止
            if topi.squeeze().item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(french_index2word[topi.item()])

            # 将本次预测的索引赋值给 input_y，进行下一个时间步预测
            input_y = topi.detach()

        # 返回结果decoded_words， 注意力张量(把没有用到的部分切掉)
        return decoded_words, decoder_attentions[:idx + 1]


# 加载模型
PATH1 = './gpumodel/my_encoderrnn.pth'
PATH2 = './gpumodel/my_attndecoderrnn.pth'
def dm06_test_Seq2Seq_Evaluate():
    # 实例化dataset对象
    mypairsdataset = MyPairsDataset(my_pairs)
    # 实例化dataloader
    mydataloader = DataLoader(dataset=mypairsdataset, batch_size=1, shuffle=True)

    # 实例化模型
    input_size = english_word_n
    hidden_size = 256  # 观察结果数据 可使用8
    my_encoderrnn = EncoderRNN(input_size, hidden_size)
    # my_encoderrnn.load_state_dict(torch.load(PATH1))
    my_encoderrnn.load_state_dict(torch.load(PATH1, map_location=lambda storage, loc: storage), False)
    print('my_encoderrnn模型结构--->', my_encoderrnn)

    # 实例化模型
    input_size = french_word_n
    hidden_size = 256  # 观察结果数据 可使用8
    my_attndecoderrnn = AttnDecoderRNN(input_size, hidden_size)
    # my_attndecoderrnn.load_state_dict(torch.load(PATH2))
    my_attndecoderrnn.load_state_dict(torch.load(PATH2, map_location=lambda storage, loc: storage), False)
    print('my_decoderrnn模型结构--->', my_attndecoderrnn)

    my_samplepairs = [['i m impressed with your french .', 'je suis impressionne par votre francais .'],
                      ['i m more than a friend .', 'je suis plus qu une amie .'],
                      ['she is beautiful like her mother .', 'vous gagnez n est ce pas ?']]
    print('my_samplepairs--->', len(my_samplepairs))

    for index, pair in enumerate(my_samplepairs):
        x = pair[0]
        y = pair[1]

        # 样本x 文本数值化
        tmpx = [english_word2index[word] for word in x.split(' ')]
        tmpx.append(EOS_token)
        tensor_x = torch.tensor(tmpx, dtype=torch.long, device=device).view(1, -1)

        # 模型预测
        decoded_words, attentions = Seq2Seq_Evaluate(tensor_x, my_encoderrnn, my_attndecoderrnn)
        # print('decoded_words->', decoded_words)
        output_sentence = ' '.join(decoded_words)

        print('\n')
        print('英文>', x)
        print('参考译文=', y)
        print('预测译文<', output_sentence)


def dm07_test_Attention():

    # 实例化dataset对象
    mypairsdataset = MyPairsDataset(my_pairs)
    # 实例化dataloader
    mydataloader = DataLoader(dataset=mypairsdataset, batch_size=1, shuffle=True)

    # 实例化模型
    input_size = english_word_n
    hidden_size = 256  # 观察结果数据 可使用8
    my_encoderrnn = EncoderRNN(input_size, hidden_size)
    # my_encoderrnn.load_state_dict(torch.load(PATH1))
    my_encoderrnn.load_state_dict(torch.load(PATH1, map_location=lambda storage, loc: storage), False)

    # 实例化模型
    input_size = french_word_n
    hidden_size = 256  # 观察结果数据 可使用8
    my_attndecoderrnn = AttnDecoderRNN(input_size, hidden_size)
    # my_attndecoderrnn.load_state_dict(torch.load(PATH2))
    my_attndecoderrnn.load_state_dict(torch.load(PATH2, map_location=lambda storage, loc: storage), False)

    sentence = "we re both teachers ."
    # 样本x 文本数值化
    tmpx = [english_word2index[word] for word in sentence.split(' ')]
    tmpx.append(EOS_token)
    tensor_x = torch.tensor(tmpx, dtype=torch.long, device=device).view(1, -1)

    # 模型预测
    decoded_words, attentions = Seq2Seq_Evaluate(tensor_x, my_encoderrnn, my_attndecoderrnn)
    print('decoded_words->', decoded_words)

    # print('\n')
    # print('英文', sentence)
    # print('法文', output_sentence)

    plt.matshow(attentions.numpy()) # 以矩阵列表的形式 显示
    # 保存图像
    plt.savefig("./s2s_attn.png")
    plt.show()

    print('attentions.numpy()--->\n', attentions.numpy())
    print('attentions.size--->', attentions.size())

    pass

if __name__ == '__main__':
    dm06_test_Seq2Seq_Evaluate()
    # dm07_test_Attention()
    print('模型评估 End')

