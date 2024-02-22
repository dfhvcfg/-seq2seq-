# 基于注意力机制的seq2seq，用于英翻法
# 1 seq2seq介绍
## 1.1 seq2seq模型架构

seq2seq模型架构分析:
![image](https://github.com/dfhvcfg/-seq2seq-/assets/57213191/96c4ee81-a306-4a32-bb73-520d4a88b6ef)

seq2seq模型架构包括三部分，分别是encoder(编码器)、decoder(解码器)、中间语义张量c。其中编码器和解码器的内部实现都使用了GRU模型
图中表示的是一个中文到英文的翻译：欢迎 来 北京 → welcome to BeiJing。编码器首先处理中文输入"欢迎 来 北京"，通过GRU模型获得每个时间步的输出张量，最后将它们拼接成一个中间语义张量c；接着解码器将使用这个中间语义张量c以及每一个时间步的隐层张量, 逐个生成对应的翻译语言
我们的案例通过英译法来讲解seq2seq设计与实现。

# 2 数据集介绍

i am from brazil .  je viens du bresil .
i am from france .  je viens de france .
i am from russia .  je viens de russie .
i am frying fish .  je fais frire du poisson .
i am not kidding .  je ne blague pas .
i am on duty now .  maintenant je suis en service .
i am on duty now .  je suis actuellement en service .
i am only joking .  je ne fais que blaguer .
i am out of time .  je suis a court de temps .
i am out of work .  je suis au chomage .
i am out of work .  je suis sans travail .
i am paid weekly .  je suis payee a la semaine .
i am pretty sure .  je suis relativement sur .
i am truly sorry .  je suis vraiment desole .
i am truly sorry .  je suis vraiment desolee .

# 3 案例步骤
基于GRU的seq2seq模型架构实现翻译的过程:

第一步: 导入工具包和工具函数
第二步: 对持久化文件中数据进行处理, 以满足模型训练要求
第三步: 构建基于GRU的编码器和解码器
第四步: 构建模型训练函数, 并进行训练
第五步: 构建模型评估函数, 并进行测试以及Attention效果分析
