import argparse
import os
import torch

from exp.exp_informer import Exp_Informer

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')
# 使用的网络结构（方便对比实验），使用defalut更改网络结构
parser.add_argument('--model', type=str, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

# 读的数据是什么（类型 路径）
parser.add_argument('--data', type=str, default='custom', help='data')
parser.add_argument('--root_path', type=str, default='./data', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='new.csv', help='data file')
# 预测的种类及方法，M多变量预测多变量，MS多变量预测单变量，S单变量预测单变量
parser.add_argument('--features', type=str, default='S', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
# 哪一列要当作是标签,如果features=M该值则没有意义
parser.add_argument('--target', type=str, default='xlalel-true', help='target feature in S or MS task')
# 数据中存在时间 时间是以什么为单位（属于数据挖掘中的重采样）
parser.add_argument('--freq', type=str, default='s', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
# 模型最后保存位置
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
'''
影响精度的三个参数
'''
# 当前输入序列长度（可自定义）,计算attention时每个batch以96行作为最小单位
parser.add_argument('--seq_len', type=int, default=72, help='input sequence length of Informer encoder')
# 标签（带预测值的那个东西）长度（可自定义），有标签预测序列长度,label_len小于seq_len
parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')
# 预测未来序列长度 （可自定义）,预测未来多少个时间点的数据，无标签预测序列长度，通过前label_len个真实值辅助decoder进行预测pred_len个预测值
parser.add_argument('--pred_len', type=int, default=75, help='prediction sequence length')

# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]
# 编码器、解码器输入维度，你数据有多少列,要减去时间那一列。！！！注意！！！：数据中列数不能太多
parser.add_argument('--enc_in', type=int, default=18, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=18, help='decoder input size')
# 输出预测未来多少个值，如果你的features填写的是M那么和上面就一样，如果填写的MS那么这里要输入1因为你的输出只有一列数据
parser.add_argument('--c_out', type=int, default=1, help='output size')
# 隐层特征，enc和dec输出维度，数据中列数不能大于该值，必须是偶数      初始512，最优912,但是内存占用过高，选择次优624
parser.add_argument('--d_model', type=int, default=624, help='dimension of model')
# 多头注意力机制，头越多注意力越好      初始8，最优14
parser.add_argument('--n_heads', type=int, default=10, help='num of heads')
# 堆叠几层enc和dec  初始2,1  ，整体最优2,1， 2,2在一开始拟合效果特别好，最差4,2
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
# 堆叠几层encoder
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
#全连接层（多层感知机）输出维度      初始2048（512*4），越大后面拟合度越小
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
# 对Q进行采样，对Q采样的因子数，factor=5最优
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
# 数据填充
parser.add_argument('--padding', type=int, default=0, help='padding type')
# 是否下采样操作pooling
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
# 防止过拟合数据丢弃的概率
parser.add_argument('--dropout', type=float, default=0, help='dropout')
# 注意力机制
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
# 时间特征的编码方式
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
# 激活函数
parser.add_argument('--activation', type=str, default='gelu',help='activation')
# 是否在编码器中输出注意力
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
# 是否执行predict函数
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data', default=True)
# 在生成式解码器中是否使用混合注意力
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
# 读数据列
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
# windows用户只能给0
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
# 实验运行次数
parser.add_argument('--itr', type=int, default=1, help='experiments times')
# 训练轮数epoch,一次epoch即为完整的数据集通过一次神经网络训练              初始6，比8，10优，在训练集上过拟合
parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
# 将完整的数据集分成若干个batch，一次输入样本的数量就是batch_size，越大梯度越准确               初始32，比24，48优
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
# 停止策略，如果多少个epoch损失没有改变就停止训练
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
# 学习率
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
# 实验描述
parser.add_argument('--des', type=str, default='test',help='exp description')
# 损失函数
parser.add_argument('--loss', type=str, default='mse',help='loss function')
# 学习率的调整方式
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
# 是否为分布式
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

#反归一化
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=True)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
# 如果为分布式指定有几个显卡
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')
#是否仅调用模型执行预测
parser.add_argument('--pred_path_only', type=str, default=None,help='do_pred_only')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

#定义数据文件字典
data_parser = {
    'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'WTH': {'data': 'WTH.csv', 'T': 'WetBulbCelsius', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1]},
    'ECL': {'data': 'ECL.csv', 'T': 'MT_320', 'M': [321, 321, 321], 'S': [1, 1, 1], 'MS': [321, 321, 1]},
    'electricity_cleaned': {'data': 'electricity_cleaned.csv', 'T': 'Hog_office_Denita', 'M': [270, 270, 270], 'S': [1, 1, 1], 'MS': [270, 270, 1]},#逗号不能去？
    'ECL_Rat': {'data': 'ECL_Rat.csv', 'T': 'Rat_lodging_Christine', 'M': [269, 269, 269], 'S': [1, 1, 1], 'MS': [269, 269, 1]},
    'ECL_Fox': {'data': 'ECL_Fox.csv', 'T': 'Fox_education_Jaclyn', 'M': [7, 7, 1], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'custom': {'data': 'new.csv', 'T': 'xlabel-true', 'M': [18, 18, 18], 'S': [1, 1, 1], 'MS': [18, 18, 1]},
}
#将data_parser中的信息通过args.data筛选读入data_info
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

#指定循环多少个encoder
args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

#创建训练类
Exp = Exp_Informer

#将我们的数据储存进去开始训练，for循环迭代我们的训练以及预测的过程
for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features,
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor,
                args.embed, args.distil, args.mix, args.des, ii)

    exp = Exp(args) # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    #训练函数,70%用于训练，10%用于验证在训练过程不断调整模型
    exp.train(setting)
    #测试，对最后20%测试集进行测试，生成测试集最开始pred_len时间序列内对target标签列的真实值true.npy和预测值pred.npy和误差metrics.npy
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)
    #预测，对target标签列的预测值real_prediction.npy
    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)

    torch.cuda.empty_cache()
