#!/usr/bin/env python
# encoding: utf-8

### tensorflow = 1.0.0

import numpy as np
import tensorflow as tf
import random
import sys



class Seq2SeqModel(object):
    # 构建模型
    def __init__(self,
               source_vocab_size,  # 问句词汇表大小
               target_vocab_size,  # 答句词汇表大小
               buckets,
               size,    # 每层神经元数量
               num_layers,   # 模型层数
               max_gradient_norm,  # 梯度被削减到最大规范
               batch_size,   # 批次大小
               learning_rate,  # 学习速率
               learning_rate_decay_factor,  # 调整学习速率
               use_lstm=False,  # if true, we use LSTM cells instead of GRU cells
               num_samples=512,  # number of samples for sampled softmax
               forward_only=False,  # if set, we do not construct the backward pass in the model.是否仅构建前向传播
               dtype=tf.float32):
        # 参数
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        # # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        # 如果样本量比词汇表量小，用抽样softmax --- 子集中word的选取采用的是均匀采样的策略
        # sampled softmax正是在进行softmax的时候对词典进行抽样，遍历抽样所得的词典子集计算条件概率再进行softmax。
        # 这个抽样并不是简单随机抽样，至少包含当前mini-batch的样本中的词。
        # sampled softmax只在模型训练过程中使用，在模型预测时依然需要遍历词典 full softmax
        if num_samples > 0 and num_samples < self.target_vocab_size:
            w_t = tf.get_variable("proj_w", [self.target_vocab_size, size], dtype=dtype)
            w = tf.transpose(w_t)
            b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
            output_projection = (w, b)
            def sampled_loss(labels, inputs):
                labels = tf.reshape(labels, [-1, 1])
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(inputs, tf.float32)
                return tf.cast(tf.nn.sampled_softmax_loss(weights=local_w_t,biases=local_b,labels=labels,inputs=local_inputs,
                                                          num_sampled=num_samples,num_classes=self.target_vocab_size),dtype)
            softmax_loss_function = sampled_loss

        def single_cell():
            return tf.contrib.rnn.GRUCell(size)
        if use_lstm:
            def single_cell():
                return tf.contrib.rnn.BasicLSTMCell(size)
        # if use_lstm=True, 使用lstm cell, 不然使用GRU cell
        cell = single_cell()
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
        # attention模型 --- ?? detail
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(encoder_inputs,decoder_inputs,cell,
                                                                         num_encoder_symbols=source_vocab_size,
                                                                         num_decoder_symbols=target_vocab_size,
                                                                         embedding_size=size,
                                                                         output_projection=output_projection,
                                                                         feed_previous=do_decode,dtype=dtype)
        # 给模型填充数据
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in range(buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))
        for i in range(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(dtype, shape=[None],
                                                name="weight{0}".format(i)))
        # targets值是解码器偏移1位
        targets = [self.decoder_inputs[i + 1] for i in range(len(self.decoder_inputs) - 1)]
        
        # 训练模型输出
        if forward_only:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
                softmax_loss_function=softmax_loss_function)
            if output_projection is not None:
                for b in range(len(buckets)):
                    self.outputs[b] = [tf.matmul(output, output_projection[0]) + output_projection[1]
                                       for output in self.outputs[b]]
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets,
                lambda x, y: seq2seq_f(x, y, False),
                softmax_loss_function=softmax_loss_function)  # loss function -- 交叉熵损失函数
        # 训练模型，更新梯度
        params = tf.trainable_variables()
        if not forward_only: # 只有训练阶段才需要计算梯度和参数更新
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)  # 梯度算法 -- 梯度下降算法
            for b in range(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)#计算损失函数关于参数的梯度
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         max_gradient_norm) # clip gradients 防止梯度爆炸
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)) #更新参数
        # 由于当前定义了length(buckets)个graph，故返回值self.updates是一个列表对象，尺寸为length(buckets)，列表中第i个元素表示graph{i}的梯度更新操作
        self.saver = tf.train.Saver(tf.global_variables())

    def step(self, session, encoder_inputs, decoder_inputs, target_weights,bucket_id, forward_only):
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))
        # 输入填充
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)
         # 输出填充：与是否有后向传播有关
        if not forward_only:
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
        else:
            output_feed = [self.losses[bucket_id]]  # Loss for this batch.
            for l in range(decoder_size):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.
            
    # 从指定桶获取一个批次随机数据，在训练每步(step)使用
    # 将输入与输出补成同encoder_size, decoder_size大小一样的尺寸
    # weight:对于补充的数，weight值为0，原有的为1
    def get_batch(self, data, bucket_id):
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], [] 
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in range(self.batch_size):
            ## Get a random batch of encoder and decoder inputs from bucket data
            encoder_input, decoder_input = random.choice(data[bucket_id])
            encoder_pad = [0] * (encoder_size - len(encoder_input))
            # Encoder inputs are padded and then reversed.
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))  # 前面填充0，而不是后面填充！！
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            # Decoder inputs get an extra "GO" symbol, and are padded then. 例，GO boy and gilr EOS PAD PAD, [1,7,4,3,0,0]
            decoder_inputs.append([1] + decoder_input +
                            [0] * decoder_pad_size)
         # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in range(encoder_size):
            batch_encoder_inputs.append(np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(self.batch_size)], dtype=np.int32))
        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in range(decoder_size):
            batch_decoder_inputs.append(np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(self.batch_size)], dtype=np.int32))
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
             # Create target_weights to be 0 for targets that are padding.
            for batch_idx in range(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward. 因为开头的GO字符
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == 0:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights

# tf定义了tf.app.flags，用于支持接受命令行传递参数，相当于接受argv        
# 第一个是参数名称，第二个是参数默认值，第三个是参数描述     
# 执行main函数之前首先进行flags的解析，也就是说TensorFlow通过设置flags来传递tf.app.run()所需要的参数，
# 我们可以直接在程序运行前初始化flags，也可以在运行程序的时候设置命令行参数来达到传参的目的。   
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")   # 学习率
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, 
                          "Learning rate decays by this much.")     # 学习速率下降系数
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")    # 梯度被削减到最大规范
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")   # 可调 batch size
tf.app.flags.DEFINE_integer("size", 512, "Size of each model layer.")   # 可调 -- 神经元个数
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")  # 神经网络层数
tf.app.flags.DEFINE_integer("from_vocab_size", 20000, "English vocabulary size.")   # Input词典大小
tf.app.flags.DEFINE_integer("to_vocab_size", 20000, "French vocabulary size.")  # Output词典大小
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_string("from_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("from_dev_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_dev_data", None, "Training data.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")  # 每多少次迭代存储一次模型
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")    # 精度，float精度

FLAGS = tf.app.flags.FLAGS

# buckets: a list of pairs (I, O), where I specifies maximum input length
# that will be processed in that bucket, and O specifies maximum output
# length. Training instances that have inputs longer than I or outputs
# longer than O will be pushed to the next bucket and padded accordingly.
_buckets = [(5, 5),(30, 10)]  # 可调参数

def read_data(source_path, target_path, max_size=None):
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(2)   # 以EOS结尾
                # 将input sentence 和 output sentence 按照大小逐对放入Bucket
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set

def create_model(session, forward_only):
    dtype = tf.float32
    model = Seq2SeqModel(FLAGS.from_vocab_size,FLAGS.to_vocab_size,_buckets,FLAGS.size,
                                       FLAGS.num_layers,FLAGS.max_gradient_norm,FLAGS.batch_size,
                                       FLAGS.learning_rate,FLAGS.learning_rate_decay_factor,
                                       forward_only=forward_only,dtype=dtype)
    ckpt = tf.train.get_checkpoint_state('dir')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model