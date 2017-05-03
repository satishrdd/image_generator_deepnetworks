import tensorflow as tf
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle
import os
import scipy.io
import scipy.misc
import tensorflow.contrib.slim as slim

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', "'pretrain', 'train' or 'eval'")
flags.DEFINE_string('model_save_path', 'model', "directory for saving the model")
flags.DEFINE_string('sample_save_path', 'sample', "directory for saving the sampled images")
FLAGS = flags.FLAGS

import tensorflow as tf
import tensorflow.contrib.slim as slim

class Solver(object):

    def __init__(self, model, batch_size=100, pretrain_iter=20000, train_iter=2000, sample_iter=100, 
                 svhn_dir='svhn', mnist_dir='mnist', log_dir='logs', sample_save_path='sample', 
                 model_save_path='model', pretrained_model='model/svhn_model-20000', test_model='model/dtn-1800'):
        
        self.model = model
        self.batch_size = batch_size
        self.pretrain_iter = pretrain_iter
        self.train_iter = train_iter
        self.sample_iter = sample_iter
        self.svhn_dir = svhn_dir
        self.mnist_dir = mnist_dir
        self.log_dir = log_dir
        self.sample_save_path = sample_save_path
        self.model_save_path = model_save_path
        self.pretrained_model = pretrained_model
        self.test_model = test_model
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True

    def load_svhn(self, image_dir, split='train'):
        print ('loading svhn image dataset..')
        
        if self.model.mode == 'pretrain':
            image_file = 'extra_32x32.mat' if split=='train' else 'test_32x32.mat'
        else:
            image_file = 'train_32x32.mat' if split=='train' else 'test_32x32.mat'
            
        image_dir = os.path.join(image_dir, image_file)
        svhn = scipy.io.loadmat(image_dir)
        images = np.transpose(svhn['X'], [3, 0, 1, 2]) / 127.5 - 1
        labels = svhn['y'].reshape(-1)
        labels[np.where(labels==10)] = 0
        print ('finished loading svhn image dataset..!')
        return images, labels

    def load_mnist(self, image_dir, split='train'):
        print ('loading mnist image dataset..')
        image_file = 'train.pkl' if split=='train' else 'test.pkl'
        image_dir = os.path.join(image_dir, image_file)
        with open(image_dir, 'rb') as f:
            mnist = pickle.load(f)
        images = mnist['X'] / 127.5 - 1
        labels = mnist['y']
        print ('finished loading mnist image dataset..!')
        return images, labels

    def merge_images(self, sources, targets, k=10):
        _, h, w, _ = sources.shape
        row = int(np.sqrt(self.batch_size))
        merged = np.zeros([row*h, row*w*2, 3])

        for idx, (s, t) in enumerate(zip(sources, targets)):
            i = idx // row
            j = idx % row
            merged[i*h:(i+1)*h, (j*2)*h:(j*2+1)*h, :] = s
            merged[i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h, :] = t
        return merged

    def pretrain(self):
        # load svhn dataset
        train_images, train_labels = self.load_svhn(self.svhn_dir, split='train')
        test_images, test_labels = self.load_svhn(self.svhn_dir, split='test')

        # build a graph
        model = self.model
        model.build_model()
        
        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())

            for step in range(self.pretrain_iter+1):
                i = step % int(train_images.shape[0] / self.batch_size)
                batch_images = train_images[i*self.batch_size:(i+1)*self.batch_size]
                batch_labels = train_labels[i*self.batch_size:(i+1)*self.batch_size] 
                feed_dict = {model.images: batch_images, model.labels: batch_labels}
                sess.run(model.train_op, feed_dict) 

                if (step+1) % 10 == 0:
                    summary, l, acc = sess.run([model.summary_op, model.loss, model.accuracy], feed_dict)
                    rand_idxs = np.random.permutation(test_images.shape[0])[:self.batch_size]
                    test_acc, _ = sess.run(fetches=[model.accuracy, model.loss], 
                                           feed_dict={model.images: test_images[rand_idxs], 
                                                      model.labels: test_labels[rand_idxs]})
                    summary_writer.add_summary(summary, step)
                    print ('Step: [%d/%d] loss: [%.6f] train acc: [%.2f] test acc [%.2f]' \
                               %(step+1, self.pretrain_iter, l, acc, test_acc))

                if (step+1) % 1000 == 0:  
                    saver.save(sess, os.path.join(self.model_save_path, 'svhn_model'), global_step=step+1) 
                    print ('svhn_model-%d saved..!' %(step+1))

    def train(self):
        # load svhn dataset
        svhn_images, _ = self.load_svhn(self.svhn_dir, split='train')
        mnist_images, _ = self.load_mnist(self.mnist_dir, split='train')

        # build a graph
        model = self.model
        model.build_model()

        # make directory if not exists
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)

        with tf.Session(config=self.config) as sess:
            # initialize G and D
            tf.global_variables_initializer().run()
            # restore variables of F
            print ('loading pretrained model F..')
            variables_to_restore = slim.get_model_variables(scope='content_extractor')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_model)
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver()

            print ('start training..!')
            f_interval = 15
            for step in range(self.train_iter+1):
                
                i = step % int(svhn_images.shape[0] / self.batch_size)
                # train the model for source domain S
                src_images = svhn_images[i*self.batch_size:(i+1)*self.batch_size]
                feed_dict = {model.src_images: src_images}
                
                sess.run(model.d_train_op_src, feed_dict) 
                sess.run([model.g_train_op_src], feed_dict)
                sess.run([model.g_train_op_src], feed_dict) 
                sess.run([model.g_train_op_src], feed_dict) 
                sess.run([model.g_train_op_src], feed_dict) 
                sess.run([model.g_train_op_src], feed_dict) 
                sess.run([model.g_train_op_src], feed_dict)
                
                if step > 1600:
                    f_interval = 30
                
                if i % f_interval == 0:
                    sess.run(model.f_train_op_src, feed_dict)
                
                if (step+1) % 10 == 0:
                    summary, dl, gl, fl = sess.run([model.summary_op_src, \
                        model.d_loss_src, model.g_loss_src, model.f_loss_src], feed_dict)
                    summary_writer.add_summary(summary, step)
                    print ('[Source] step: [%d/%d] d_loss: [%.6f] g_loss: [%.6f] f_loss: [%.6f]' \
                               %(step+1, self.train_iter, dl, gl, fl))
                
                # train the model for target domain T
                j = step % int(mnist_images.shape[0] / self.batch_size)
                trg_images = mnist_images[j*self.batch_size:(j+1)*self.batch_size]
                feed_dict = {model.src_images: src_images, model.trg_images: trg_images}
                sess.run(model.d_train_op_trg, feed_dict)
                sess.run(model.d_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)

                if (step+1) % 10 == 0:
                    summary, dl, gl = sess.run([model.summary_op_trg, \
                        model.d_loss_trg, model.g_loss_trg], feed_dict)
                    summary_writer.add_summary(summary, step)
                    print ('[Target] step: [%d/%d] d_loss: [%.6f] g_loss: [%.6f]' \
                               %(step+1, self.train_iter, dl, gl))

                if (step+1) % 200 == 0:
                    saver.save(sess, os.path.join(self.model_save_path, 'dtn'), global_step=step+1)
                    print ('model/dtn-%d saved' %(step+1))
                
    def eval(self):
        # build model
        model = self.model
        model.build_model()

        # load svhn dataset
        svhn_images, _ = self.load_svhn(self.svhn_dir)

        with tf.Session(config=self.config) as sess:
            # load trained parameters
            print ('loading test model..')
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)

            print ('start sampling..!')
            for i in range(self.sample_iter):
                # train model for source domain S
                batch_images = svhn_images[i*self.batch_size:(i+1)*self.batch_size]
                feed_dict = {model.images: batch_images}
                sampled_batch_images = sess.run(model.sampled_images, feed_dict)

                # merge and save source images and sampled target images
                merged = self.merge_images(batch_images, sampled_batch_images)
                path = os.path.join(self.sample_save_path, 'sample-%d-to-%d.png' %(i*self.batch_size, (i+1)*self.batch_size))
                scipy.misc.imsave(path, merged)
                print ('saved %s' %path)

class DTN(object):
    """Domain Transfer Network
    """
    def __init__(self, mode='train', learning_rate=0.0003):
        self.mode = mode
        self.learning_rate = learning_rate
        
    def content_extractor(self, images, reuse=False):
        # images: (batch, 32, 32, 3) or (batch, 32, 32, 1)
        
        if images.get_shape()[3] == 1:
            # For mnist dataset, replicate the gray scale image 3 times.
            images = tf.image.grayscale_to_rgb(images)
        
        with tf.variable_scope('content_extractor', reuse=reuse):
            with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=None,
                                 stride=2,  weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                    activation_fn=tf.nn.relu, is_training=(self.mode=='train' or self.mode=='pretrain')):
                    
                    net = slim.conv2d(images, 32, [3, 3], scope='conv1')   # (batch_size, 16, 16, 64)
                    net = slim.batch_norm(net, scope='bn1')
                    net = slim.conv2d(net,64, [3, 3], scope='conv2')     # (batch_size, 8, 8, 128)
                    net = slim.batch_norm(net, scope='bn2')
                    net = slim.conv2d(net, 128, [3, 3], scope='conv3')     # (batch_size, 4, 4, 256)
                    net = slim.batch_norm(net, scope='bn3')
                    net = slim.conv2d(net, 256, [4, 4], padding='VALID', scope='conv4')   # (batch_size, 1, 1, 128)
                    net = slim.batch_norm(net, activation_fn=tf.nn.tanh, scope='bn4')
                    if self.mode == 'pretrain':
                        net = slim.conv2d(net, 10, [1, 1], padding='VALID', scope='out')
                        net = slim.flatten(net)
                    return net
                
    def generator(self, inputs, reuse=False):
        # inputs: (batch, 1, 1, 128)
        with tf.variable_scope('generator', reuse=reuse):
            with slim.arg_scope([slim.conv2d_transpose], padding='SAME', activation_fn=None,           
                                 stride=2, weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                     activation_fn=tf.nn.relu, is_training=(self.mode=='train')):

                    net = slim.conv2d_transpose(inputs, 512, [4, 4], padding='VALID', scope='conv_transpose1')   # (batch_size, 4, 4, 512)
                    net = slim.batch_norm(net, scope='bn1')
                    net = slim.conv2d_transpose(net, 256, [3, 3], scope='conv_transpose2')  # (batch_size, 8, 8, 256)
                    net = slim.batch_norm(net, scope='bn2')
                    net = slim.conv2d_transpose(net, 128, [3, 3], scope='conv_transpose3')  # (batch_size, 16, 16, 128)
                    net = slim.batch_norm(net, scope='bn3')
                    net = slim.conv2d_transpose(net, 1, [3, 3], activation_fn=tf.nn.tanh, scope='conv_transpose4')   # (batch_size, 32, 32, 1)
                    return net
    
    def discriminator(self, images, reuse=False):
        # images: (batch, 32, 32, 1)
        with tf.variable_scope('discriminator', reuse=reuse):
            with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=None,
                                 stride=2,  weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                    activation_fn=tf.nn.relu, is_training=(self.mode=='train')):
                    
                    net = slim.conv2d(images, 128, [3, 3], activation_fn=tf.nn.relu, scope='conv1')   # (batch_size, 16, 16, 128)
                    net = slim.batch_norm(net, scope='bn1')
                    net = slim.conv2d(net, 256, [3, 3], scope='conv2')   # (batch_size, 8, 8, 256)
                    net = slim.batch_norm(net, scope='bn2')
                    net = slim.conv2d(net, 512, [3, 3], scope='conv3')   # (batch_size, 4, 4, 512)
                    net = slim.batch_norm(net, scope='bn3')
                    net = slim.conv2d(net, 1, [4, 4], padding='VALID', scope='conv4')   # (batch_size, 1, 1, 1)
                    net = slim.flatten(net)
                    return net
                
    def build_model(self):
        
        if self.mode == 'pretrain':
            self.images = tf.placeholder(tf.float32, [None, 32, 32, 3], 'svhn_images')
            self.labels = tf.placeholder(tf.int64, [None], 'svhn_labels')
            
            # logits and accuracy
            self.logits = self.content_extractor(self.images)
            self.pred = tf.argmax(self.logits, 1)
            self.correct_pred = tf.equal(self.pred, self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

            # loss and train op
            self.loss = slim.losses.sparse_softmax_cross_entropy(self.logits, self.labels)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate) 
            self.train_op = slim.learning.create_train_op(self.loss, self.optimizer)
            
            # summary op
            loss_summary = tf.summary.scalar('classification_loss', self.loss)
            accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
            self.summary_op = tf.summary.merge([loss_summary, accuracy_summary])

        elif self.mode == 'eval':
            self.images = tf.placeholder(tf.float32, [None, 32, 32, 3], 'svhn_images')

            # source domain (svhn to mnist)
            self.fx = self.content_extractor(self.images)
            self.sampled_images = self.generator(self.fx)

        elif self.mode == 'train':
            self.src_images = tf.placeholder(tf.float32, [None, 32, 32, 3], 'svhn_images')
            self.trg_images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'mnist_images')
            
            # source domain (svhn to mnist)
            self.fx = self.content_extractor(self.src_images)
            self.fake_images = self.generator(self.fx)
            self.logits = self.discriminator(self.fake_images)
            self.fgfx = self.content_extractor(self.fake_images, reuse=True)

            # loss
            self.d_loss_src = slim.losses.sigmoid_cross_entropy(self.logits, tf.zeros_like(self.logits))
            self.g_loss_src = slim.losses.sigmoid_cross_entropy(self.logits, tf.ones_like(self.logits))
            self.f_loss_src = tf.reduce_mean(tf.square(self.fx - self.fgfx)) * 15.0
            
            # optimizer
            self.d_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)
            self.g_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)
            self.f_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)
            
            t_vars = tf.trainable_variables()
            d_vars = [var for var in t_vars if 'discriminator' in var.name]
            g_vars = [var for var in t_vars if 'generator' in var.name]
            f_vars = [var for var in t_vars if 'content_extractor' in var.name]
            
            # train op
            with tf.name_scope('source_train_op'):
                self.d_train_op_src = slim.learning.create_train_op(self.d_loss_src, self.d_optimizer_src, variables_to_train=d_vars)
            with tf.name_scope('source_train_op2'):
                self.g_train_op_src = slim.learning.create_train_op(self.g_loss_src, self.g_optimizer_src, variables_to_train=g_vars)
            with tf.name_scope('source_train_op3'):
                self.f_train_op_src = slim.learning.create_train_op(self.f_loss_src, self.f_optimizer_src, variables_to_train=f_vars)
            
            # summary op
            d_loss_src_summary = tf.summary.scalar('src_d_loss', self.d_loss_src)
            g_loss_src_summary = tf.summary.scalar('src_g_loss', self.g_loss_src)
            f_loss_src_summary = tf.summary.scalar('src_f_loss', self.f_loss_src)
            origin_images_summary = tf.summary.image('src_origin_images', self.src_images)
            sampled_images_summary = tf.summary.image('src_sampled_images', self.fake_images)
            self.summary_op_src = tf.summary.merge([d_loss_src_summary, g_loss_src_summary, 
                                                    f_loss_src_summary, origin_images_summary, 
                                                    sampled_images_summary])
            
            # target domain (mnist)
            self.fx = self.content_extractor(self.trg_images, reuse=True)
            self.reconst_images = self.generator(self.fx, reuse=True)
            self.logits_fake = self.discriminator(self.reconst_images, reuse=True)
            self.logits_real = self.discriminator(self.trg_images, reuse=True)
            
            # loss
            self.d_loss_fake_trg = slim.losses.sigmoid_cross_entropy(self.logits_fake, tf.zeros_like(self.logits_fake))
            self.d_loss_real_trg = slim.losses.sigmoid_cross_entropy(self.logits_real, tf.ones_like(self.logits_real))
            self.d_loss_trg = self.d_loss_fake_trg + self.d_loss_real_trg
            self.g_loss_fake_trg = slim.losses.sigmoid_cross_entropy(self.logits_fake, tf.ones_like(self.logits_fake))
            self.g_loss_const_trg = tf.reduce_mean(tf.square(self.trg_images - self.reconst_images)) * 15.0
            self.g_loss_trg = self.g_loss_fake_trg + self.g_loss_const_trg
            
            # optimizer
            self.d_optimizer_trg = tf.train.AdamOptimizer(self.learning_rate)
            self.g_optimizer_trg = tf.train.AdamOptimizer(self.learning_rate)

            # train op
            with tf.name_scope('target_train_op'):
                self.d_train_op_trg = slim.learning.create_train_op(self.d_loss_trg, self.d_optimizer_trg, variables_to_train=d_vars)
                self.g_train_op_trg = slim.learning.create_train_op(self.g_loss_trg, self.g_optimizer_trg, variables_to_train=g_vars)
            
            # summary op
            d_loss_fake_trg_summary = tf.summary.scalar('trg_d_loss_fake', self.d_loss_fake_trg)
            d_loss_real_trg_summary = tf.summary.scalar('trg_d_loss_real', self.d_loss_real_trg)
            d_loss_trg_summary = tf.summary.scalar('trg_d_loss', self.d_loss_trg)
            g_loss_fake_trg_summary = tf.summary.scalar('trg_g_loss_fake', self.g_loss_fake_trg)
            g_loss_const_trg_summary = tf.summary.scalar('trg_g_loss_const', self.g_loss_const_trg)
            g_loss_trg_summary = tf.summary.scalar('trg_g_loss', self.g_loss_trg)
            origin_images_summary = tf.summary.image('trg_origin_images', self.trg_images)
            sampled_images_summary = tf.summary.image('trg_reconstructed_images', self.reconst_images)
            self.summary_op_trg = tf.summary.merge([d_loss_trg_summary, g_loss_trg_summary, 
                                                    d_loss_fake_trg_summary, d_loss_real_trg_summary,
                                                    g_loss_fake_trg_summary, g_loss_const_trg_summary,
                                                    origin_images_summary, sampled_images_summary])
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)


def main(_):
    
    model = DTN(mode=FLAGS.mode, learning_rate=0.0003)
    solver = Solver(model, batch_size=100, pretrain_iter=20000, train_iter=2000, sample_iter=100, 
                    svhn_dir='svhn', mnist_dir='mnist', model_save_path=FLAGS.model_save_path, sample_save_path=FLAGS.sample_save_path)
    
    # create directories if not exist
    if not tf.gfile.Exists(FLAGS.model_save_path):
        tf.gfile.MakeDirs(FLAGS.model_save_path)
    if not tf.gfile.Exists(FLAGS.sample_save_path):
        tf.gfile.MakeDirs(FLAGS.sample_save_path)
    
    if FLAGS.mode == 'pretrain':
        solver.pretrain()
    elif FLAGS.mode == 'train':
        solver.train()
    else:
        solver.eval()
        
if __name__ == '__main__':
    tf.app.run()