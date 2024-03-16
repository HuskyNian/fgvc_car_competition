import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import numpy as np
from sklearn.metrics import average_precision_score
import os
import logging
import traceback
import datetime
import json

tf.get_logger().setLevel('ERROR')

logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s]\t%(message)s')


def export_tf(sess, export_dir, model_name, input_names, output_names):
    output_graph_def = convert_variables_to_constants(sess,
                                                      sess.graph_def,
                                                      output_names
                                                      )
    tf.train.write_graph(output_graph_def, export_dir, model_name, as_text=False)

    input_names_new = [i + ':0' for i in input_names]
    output_names_new = [o + ':0' for o in output_names]
    meta = {
        "input_names": input_names_new,
        "output_names": output_names_new
    }

    with open(os.path.join(export_dir, "graph_meta.json"), "w") as f:
        f.write(json.dumps(meta))


def train_sample_op(train_path, input_dim, nClasses, batch_size, num_epochs):
    '''返回训练样本及labels'''
    feat_defaults = [','.join(['0' for _ in range(input_dim)])]
    record_defaults = [['-1'], ['-1'], ['-1'], ['-1'], feat_defaults, ['0']]

    file_queue = tf.train.string_input_producer([train_path], num_epochs=num_epochs)
    reader = tf.TextLineReader()

    keys, values = reader.read_up_to(file_queue, num_records=batch_size)
    batch_values = tf.train.batch([values], batch_size=batch_size, capacity=64000, enqueue_many=True, num_threads=16, allow_smaller_final_batch=True)
    datas = tf.decode_csv(batch_values, record_defaults=record_defaults, field_delim=',')

    biz_id = datas[0]
    usr_id = datas[1]
    itm_id = datas[2]
    vis_t  = datas[3]
    defaults = [['0'] for _ in range(input_dim)]
    features = tf.decode_csv(datas[4], record_defaults=defaults, field_delim=' ')
    features = tf.string_to_number(features)
    features = tf.transpose(features) # now, features with shape: batch_size * feat_num

    label = tf.string_to_number(datas[5], tf.int32)
    one_hot_label = tf.one_hot(indices=label, depth=nClasses)

    return biz_id, features, label, one_hot_label


## model op
def build_model(batch_size, input_dim, nClasses, hidden_layers, reg_weight, lr):
    tf.train.get_or_create_global_step()
    x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='input')
    one_hot_label = tf.placeholder(dtype=tf.float32, shape=[None, nClasses])
    layers = [x]
    weights = {}
    biases = {}
    in_d = input_dim
    for n in range(len(hidden_layers)):
        out_d = hidden_layers[n]
        tmp_name = 'l_' + str(n)
        weights[tmp_name] = tf.Variable(initial_value=tf.truncated_normal(shape=[in_d, out_d]),
                                        name='weights_' + str(n))
        tf.summary.histogram('weights_' + str(n), weights[tmp_name])
        biases[tmp_name] = tf.Variable(initial_value=tf.zeros(shape=[out_d]), name='biases_' + str(n))
        tf.summary.histogram('biases_' + str(n), biases[tmp_name])
        in_d = out_d
    weights['out'] = tf.Variable(initial_value=tf.truncated_normal(shape=[in_d, nClasses]), name='weights_out')
    tf.summary.histogram('weights_out', weights['out'])
    biases['out'] = tf.Variable(initial_value=tf.zeros(shape=[nClasses]), name='biases_out')
    tf.summary.histogram('biases_out', biases['out'])
    for n in range(len(hidden_layers)):
        tmp_name = 'l_' + str(n)
        layers.append(tf.nn.tanh(tf.matmul(layers[-1], weights[tmp_name])))

    logit = tf.matmul(layers[-1], weights['out']) + biases['out']
    y = tf.nn.softmax(logit, name='output')
    pred_label = tf.cast(y[:, 1] > 0.5, tf.int32, name='pred_label')

    soft_loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_label, logits=logit)
    regular_loss = tf.losses.get_regularization_loss()
    loss = soft_loss + reg_weight * regular_loss
    tf.summary.scalar('loss', loss)
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    return x, one_hot_label, y, loss, train_op, pred_label

#####
def train(train_path, model_dir, save_name):
    hidden_layers=[64,128,64]
    nClasses=2
    reg_weight=0.01
    num_epochs=2
    input_dim=75
    lr=0.0005
    batch_size=1000

    print_info('building model graph ...')
    model_graph = tf.Graph()
    with model_graph.as_default():
        model_ops = build_model(batch_size, input_dim, nClasses, hidden_layers, reg_weight, lr)
        x_ph, one_hot_label_ph, pred_op, loss_op, train_op, pred_label  = model_ops

        model_sess = tf.Session()
        model_sess.run(tf.global_variables_initializer())
        model_sess.run(tf.local_variables_initializer())
        merged = tf.summary.merge_all() #将图形、训练过程等数据合并在一起
        writer = tf.summary.FileWriter(model_dir + '/summary', model_sess.graph) #将训练日志写入到logs文件夹下

    print_info('building read graph for training ...')
    read_graph = tf.Graph()
    with read_graph.as_default():
        trn_datas = train_sample_op(train_path, input_dim, nClasses, batch_size, num_epochs)
        trn_id, trn_samples, trn_labels, trn_one_hot_label = trn_datas

        read_sess = tf.Session()
        read_sess.run(tf.global_variables_initializer())
        read_sess.run(tf.local_variables_initializer())
        read_coord = tf.train.Coordinator()
        read_threads = tf.train.start_queue_runners(coord=read_coord, sess=read_sess)

    print_info('training ...')
    try:

        iteration = 0
        while True:
            iteration += 1
            train_sample_, train_label_, train_one_hot_label_ = read_sess.run([trn_samples, trn_labels, trn_one_hot_label])
            # print train_label_
            # print train_one_hot_label_
            model_sess.run(train_op,feed_dict={x_ph: train_sample_, one_hot_label_ph: train_one_hot_label_})

            if iteration % 500 == 0:
                lo = model_sess.run(loss_op, feed_dict={x_ph: train_sample_, one_hot_label_ph: train_one_hot_label_})

                print_info('')
                print_info('step: %d, loss: %.4f' % (iteration, lo))
                print_info('')

                train_pred_ = model_sess.run(pred_op, feed_dict={x_ph: train_sample_, one_hot_label_ph: train_one_hot_label_})

                print_info('train set pred result: ..............................')
                for c in range(nClasses):
                    y_test = train_one_hot_label_[:, c]
                    y_score = train_pred_[:, c]
                    pos_num = np.sum(y_test)
                    neg_num = y_test.shape[0] - pos_num
                    ratio = 1.0 * pos_num / neg_num

                    AP, prs = metrics(y_test, y_score)
                    msg = 'label: {:d};\ttotal: {:d};\t#pos: {:d};\t#neg: {:d};\tratio: {:.3f};\tAP: {:.3f}'
                    print_info(msg.format(c, y_test.shape[0], int(pos_num), int(neg_num), ratio, AP))
                    print_info('(threshold, precision, recall)')
                    print_info(prs)

                result = model_sess.run(merged,feed_dict={x_ph: train_sample_, one_hot_label_ph: train_one_hot_label_}) #计算需要写入的日志数据
                writer.add_summary(result, iteration) #将日志数据写入文件

            if iteration % 10000 == 0:
                with model_graph.as_default():
                    saver = tf.train.Saver()
                    save_path = os.path.join(model_dir, save_name)
                    save_path = saver.save(model_sess, save_path, global_step=iteration)
                    print_info("Model saved in path: %s" % save_path)

    except tf.errors.OutOfRangeError as e :
        print_info('out of range with iteration: %d' % iteration)
    except Exception as e:
        print_err(traceback.print_exc())
    finally:
        print_info('finished training ...')
        with model_graph.as_default() as graph:
            saver = tf.train.Saver()
            save_path = os.path.join(model_dir, save_name)
            save_path = saver.save(model_sess, save_path, global_step=iteration)
            print_info("Model saved in path: %s" % save_path)

            # saved as frozen model
            input_names = ['input']
            output_names = ['pred_label']
            save_path = os.path.join(model_dir, 'frozen_model')
            export_tf(model_sess, save_path, 'frozen_inference_graph.pb', input_names, output_names)

        read_coord.request_stop()
        read_coord.join(read_threads)
        model_sess.close()
        read_sess.close()
        writer.close()

def metrics(labels, preds):
    # labels: 0/1
    assert len(labels) == len(preds)
    AP = average_precision_score(labels, preds)
    # th: 0.5, 0.6, 0.7, 0.8, 0.9
    labels = np.array(labels, int)
    preds = np.array(preds, float)
    prs = []
    try:
        for th in range(5, 10):
            th = th / 10.0
            pred_hard = np.array(preds > th, int)
            true_pos = np.sum(labels)
            pred_pos = np.sum(pred_hard)

            TP = np.sum(labels * pred_hard)
            precision = 1.0 * TP / pred_pos
            recall = 1.0 * TP / true_pos
            prs.append((th, precision, recall))
    except:
        print_err('error occur while calculating metrics')
        return AP, prs
    return AP, prs


def print_info(msg):
    time_stamp = datetime.datetime.now()
    time = time_stamp.strftime('[%Y-%m-%d %H:%M:%S INFO]\t')
    logging.info(str(msg))
    print(str(time) + str(msg))


def print_err(msg):
    time_stamp = datetime.datetime.now()
    time = time_stamp.strftime('[%Y-%m-%d %H:%M:%S ERROR]\t')
    logging.error(str(msg))
    print(str(time) + str(msg))