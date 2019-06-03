#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf
import math
import os
import sys
import time
import seq2seq_model

def train():
    with tf.Session() as sess:
        # 1.创建一个模型
        model = seq2seq_model.create_model(sess, False)  # forward_only= False 训练时false
        # 2.读入测试集
        dev_set = seq2seq_model.read_data("test_prepared.cn","test_prepared.sq")
        # 3.读入训练集
        train_set = seq2seq_model.read_data("train_prepared.cn","train_prepared.sq")
        print("data prepared")
        train_bucket_sizes = [len(train_set[b]) for b in range(len(seq2seq_model._buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in range(len(train_bucket_sizes))]
        print("buckets prepared")
        step_time = 0.0
        loss = 0.0
        current_step = 0
        previous_losses = []
        while True:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            # 随机生成一个0-1数，在生成bucket_id中使用
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])
            start_time = time.time()
            # Get a batch and make a step.
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id) # get a batch
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,target_weights, bucket_id, False)  # make a step
            step_time += (time.time() - start_time) / seq2seq_model.FLAGS.steps_per_checkpoint  # 平均一次的时间
            loss += step_loss / seq2seq_model.FLAGS.steps_per_checkpoint  # 平均loss 200个batch的Loss的平均值
            current_step += 1
            if current_step % seq2seq_model.FLAGS.steps_per_checkpoint == 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")  # 总混淆度（加权平均?)
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
                       "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
                 # 如果损失值在最近3次内没有再降低，减小学习率
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                checkpoint_path = os.path.join('dir', "translate.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                for bucket_id in range(len(seq2seq_model._buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        print("  eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(dev_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,target_weights, bucket_id, True)
                    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                    print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))   # 每个Bucket对应一个混淆度
                sys.stdout.flush()

def main(_):
    train()

if __name__ == "__main__":
  tf.app.run()
