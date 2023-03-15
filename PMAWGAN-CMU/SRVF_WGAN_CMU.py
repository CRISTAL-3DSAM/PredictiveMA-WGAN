import scipy.io
from ops_ExprGAN import *
import numpy as np
from scipy.io import loadmat
import time
import tensorflow as tf


np.random.seed(2019)


class SRVF_WGAN(object):
    def __init__(self,
                 session,
                 size_SRVF_H=51,
                 size_SRVF_W=26,
                 nb_frames=25,
                 Gram_matrix_size = 25,
                 size_kernel=5,
                 #size_batch=128,
                 size_batch=3, # taille test
                 num_encoder_channels=64,
                 num_z_channels=50,
                 num_input_channels=1,
                 y_dim=2, # pas utilise
                 rb_dim=3, # pas utilise
                 test_batch_low=0,
                 test_batch_up=0,
                 num_gen_channels=512,
                 enable_tile_label=True,
                 tile_ratio=1.0,
                 is_training=True,
                 disc_iters=4,  # For WGAN and WGAN-GP, number of descri iters per gener iter
                 is_flip=True,
                 data_dir = 'data_skeleton',
                 load_train = 'Data_skeleton/Data_train.txt',
                 load_test = 'Data_skeleton/Data_test.txt',
                 load_qmean = 'Data_skeleton/q_mean_data.mat',
                 discription='HUMAN',
                 checkpoint_dir='./checkpoint',
                 checkpoint_name = 'model-135000',
                 save_dir='Results/',
                 generated_dir='genrated_samples_long',
                 num_epochs=200,
                 learning_rate=0.0001,
                 LAMBDA=10,  # Gradient penalty lambda hyperparameter
                 coeff_skel_loss = 100,# coefficient for loss on skeleton position
                 # Skel_links = np.array([[1,2],[2,3],[3,4],[5,6],[6,7],[7,8],[1,9],[5,9],[9,10],[10,11],[11,12],[12,13],[10,14],[14,15],[15,16],[16,17],[17,18],[16,19],[10,20],[20,21],[21,22],[22,23],[23,24],[22,25]])-1, # indices of gram coefficnents corresponding to bones lenght, must be changed depending on the skeleton used
                 Skel_links=np.array([[1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [9, 12],[12, 13], [13, 14], [9, 15], [15, 16], [16, 17]]) - 1,
                 Bone_Loss_coeff = 100,
                 ):

        self.session = session
        self.size_SRVF_H = size_SRVF_H
        self.size_SRVF_W = size_SRVF_W
        self.nb_frames = nb_frames
        self.size_kernel = size_kernel
        self.Gram_matrix_size = Gram_matrix_size
        self.size_batch = size_batch
        self.num_input_channels = num_input_channels
        self.num_encoder_channels = num_encoder_channels
        self.num_z_channels = num_z_channels
        self.y_dim = y_dim
        self.rb_dim = rb_dim
        self.test_batch_low=test_batch_low
        self.test_batch_up=test_batch_up
        self.num_gen_channels = num_gen_channels
        self.enable_tile_label = enable_tile_label
        self.tile_ratio = tile_ratio
        self.is_training = is_training
        self.data_dir = data_dir
        self.load_train =load_train
        self.load_test = load_test
        self.load_qmean=load_qmean
        self.save_dir = save_dir
        self.is_flip = is_flip
        self.checkpoint_dir = checkpoint_dir + discription
        self.checkpoint_name = checkpoint_name
        self.generated_dir=generated_dir
        self.disc_iters = disc_iters
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.LAMBDA = LAMBDA
        self.discription = discription
        self.coeff_skel_loss = coeff_skel_loss
        self.Skel_links = Skel_links
        self.Bone_Loss_coeff = Bone_Loss_coeff
        print(self.discription)

        ###reference point
        Q_ref_ = loadmat(self.load_qmean)
        q_mean = Q_ref_['q_mean']
        self.Q_ref = np.zeros([self.size_batch, self.size_SRVF_H, self.size_SRVF_W])
        for i in range(self.Q_ref.shape[0]):
            self.Q_ref[i, :, :] = q_mean

        self.Q_ref_tensor = tf.constant(self.Q_ref, dtype=tf.float32)

        self.indices = []
        self.lenlinks = len(Skel_links)
        for b in range(0, self.size_batch):
            for f in range(0, self.size_SRVF_W):
                for j in range(0, self.lenlinks):
                    self.indices.append([b, f, Skel_links[j, 0], Skel_links[j, 1]])

        self.indices_L = []
        self.indices_R = []
        for b in range(0, self.size_batch):
            for f in range(0, self.size_SRVF_W):
                for d in range(0, 3):
                    for j in range(0, self.lenlinks):
                        self.indices_L.append([b, f, d, self.Skel_links[j, 0]])
                        self.indices_R.append([b, f, d, self.Skel_links[j, 1]])

        print("\n\tLoading data")
        self.data_y, self.data_X, self.first_frames, self.Joints = self.load_data(self.load_train)
        self.data_X = [os.path.join(self.data_dir, x) for x in self.data_X]
        # get path for y SRVF file
        self.data_y = [os.path.join(self.data_dir, y) for y in self.data_y]
        self.first_frames = [os.path.join(self.data_dir, f) for f in self.first_frames]
        self.Joints = [os.path.join(self.data_dir, j) for j in self.Joints]


        self.real_data = tf.compat.v1.placeholder(
            tf.float32,
            [self.size_batch, self.size_SRVF_H * self.size_SRVF_W],
            name='real_data'
        )
        self.emotion = tf.compat.v1.placeholder(
            tf.float32,
            #[self.size_batch, self.y_dim * self.rb_dim],
            [self.size_batch, self.size_SRVF_H * self.nb_frames], # changement de dimensions
            name='emotion_labels'
        )

        self.first_frames_real = tf.compat.v1.placeholder(
            tf.float32,
            [self.size_batch ,self.size_SRVF_H, self.size_SRVF_W],
            name='first_skeleton_frames'
        )

        self.Joints_pos_real = tf.compat.v1.placeholder(
            tf.float32,
            [self.size_batch, self.size_SRVF_H, self.size_SRVF_W],
            name='joint_position'
        )

        self.log_real = self.log_map(self.real_data)
        #self.fake_data, self.Lands_Gen = self.Generator(self.emotion, self.first_frames_real)
        self.fake_data = self.Generator(self.emotion, self.first_frames_real)
        self.exp_fake = self.exp_map(self.fake_data)
        self.log_exp_fake = self.log_map(self.exp_fake)
        self.disc_log_real = self.Discriminator(self.log_real, self.emotion, enable_bn=True)
        self.disc_log_exp_fake = self.Discriminator(self.log_exp_fake, self.emotion, reuse_variables=True,
                                                    enable_bn=True)

        ############### losses to minimize 
        ## reconstruction_loss = tf.nn.l2.loss(log_exp_fake - log_real)   #L2 loss
        reconstruction_loss = tf.reduce_mean(tf.abs(self.log_real - self.log_exp_fake))  # L1 loss
        Gram_loss, Bone_loss =  self.Gram_loss_func(self.exp_fake, self.Joints_pos_real,self.first_frames_real)
        self.gen_cost = -tf.reduce_mean(self.disc_log_exp_fake) + reconstruction_loss + self.coeff_skel_loss *Gram_loss + self.Bone_Loss_coeff * Bone_loss
        self.gen_cost_ = -self.gen_cost
        self.help_loss = tf.reduce_mean(self.disc_log_real)
        self.disc_cost = tf.reduce_mean(self.disc_log_exp_fake) - tf.reduce_mean(
            self.disc_log_real)  ###+ reconstruction_loss

        # penalty of improved WGAN
        alpha = tf.random.uniform(
            shape=[self.size_batch, 1],
            minval=0.,
            maxval=1.
        )
        differences = self.log_exp_fake - self.log_real
        interpolates = self.log_real + (alpha * differences)
        gradients = tf.gradients(self.Discriminator(interpolates, self.emotion, reuse_variables=True, enable_bn=True),
                                 [interpolates])[0]
        slopes = tf.sqrt(tf.compat.v1.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        self.disc_cost += self.LAMBDA * gradient_penalty

        trainable_variables = tf.compat.v1.trainable_variables()  ##returns all variables created(the two variable scopes) and makes trainable true
        self.gen_params = [var for var in trainable_variables if 'G_' in var.name]
        self.disc_params = [var for var in trainable_variables if 'D_' in var.name]

        GEN_cost_summary = tf.compat.v1.summary.scalar('GEN_cost', self.gen_cost_)
        DISC_cost_summary = tf.compat.v1.summary.scalar('DISC_cost', self.disc_cost)

        GENerator_cost_summary=tf.compat.v1.summary.scalar('GENerator_cost', -tf.reduce_mean(self.disc_log_exp_fake))
        reconstruction_cost_summary = tf.compat.v1.summary.scalar('Reconstruction_cost', reconstruction_loss)
        Gram_cost_summary =  tf.compat.v1.summary.scalar('Gram_cost', self.coeff_skel_loss * Gram_loss)
        Bone_cost_summary = tf.compat.v1.summary.scalar('Bone_cost', self.Bone_Loss_coeff * Bone_loss)


        help_cost_summary = tf.compat.v1.summary.scalar('DiscReal_cost', self.help_loss)
        self.summary = tf.compat.v1.summary.merge(
            [GEN_cost_summary, DISC_cost_summary, GENerator_cost_summary,reconstruction_cost_summary,Gram_cost_summary, Bone_cost_summary, help_cost_summary])

        self.saver = tf.compat.v1.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=5)

    def train(self,
              num_epochs=200,
              decay_rate=1.0,
              enable_shuffle=True,
              ):

        ## count number of batches seen by the graph
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        Train_learning_rate = tf.compat.v1.train.exponential_decay(
            learning_rate=self.learning_rate,
            global_step=self.global_step,
            decay_steps=1000,  ##len(self.data_X) // self.size_batch * 2,
            decay_rate=decay_rate,
            staircase=True
        )
        with tf.compat.v1.variable_scope('gen-optimize',
                                         reuse=tf.compat.v1.AUTO_REUSE):  ##tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
            self.gen_train_op = tf.compat.v1.train.AdamOptimizer(
                learning_rate=Train_learning_rate,
                beta1=0.5,
                beta2=0.9
            ).minimize(self.gen_cost, global_step=self.global_step, var_list=self.gen_params)
        with tf.compat.v1.variable_scope('disc-optimizer', reuse=tf.compat.v1.AUTO_REUSE):
            self.disc_train_op = tf.compat.v1.train.AdamOptimizer(
                learning_rate=Train_learning_rate,
                beta1=0.5,
                beta2=0.9
            ).minimize(self.disc_cost, global_step=self.global_step, var_list=self.disc_params)

        ## write summary          
        filename = 'summary' + str(self.learning_rate) + self.discription
        self.writer = tf.compat.v1.summary.FileWriter(os.path.join(self.save_dir, filename),
                                                      self.session.graph)  ##train.SummaryWriter
        try:
            tf.global_variables_initializer().run()
        except:
            tf.compat.v1.initialize_all_variables().run()
        num_batches = len(self.data_X) // self.size_batch
        for epoch in range(num_epochs):
            if enable_shuffle:
                seed = 2019
                np.random.seed(seed)
                np.random.shuffle(self.data_X)
                np.random.seed(seed)
                np.random.shuffle(self.data_y)
                np.random.seed(seed)
                np.random.shuffle(self.first_frames)
                np.random.seed(seed)
                np.random.shuffle(self.Joints)
            for ind_batch in range(num_batches):
                start_time = time.time()
                batch_files = self.data_X[ind_batch * self.size_batch:(ind_batch + 1) * self.size_batch]
                batch = [self.read_SRVF(
                    path_SRVF=batch_file) for batch_file in batch_files]
                batch_SRVF = np.array(batch).astype(np.float32)

                ## utilise les SRVF au lieu des matrices de labels
                batch_files_emo = self.data_y[ind_batch * self.size_batch:(ind_batch + 1) * self.size_batch]
                batch_emo = [self.read_SRVF(
                    path_SRVF=batch_file) for batch_file in batch_files_emo]
                batch_label_emo = np.array(batch_emo).astype(np.float32)

                batch_files_frame = self.first_frames[ind_batch * self.size_batch:(ind_batch + 1) * self.size_batch]
                batch_fframe = [self.read_first_frame(
                    path_frame=batch_file) for batch_file in batch_files_frame]
                batch_first_frame = np.array(batch_fframe).astype(np.float32)

                batch_files_frames = self.Joints[ind_batch * self.size_batch:(ind_batch + 1) * self.size_batch]
                batch_frames = [self.read_frames(
                    path=batch_file) for batch_file in batch_files_frames]
                batch_all_frames = np.array(batch_frames).astype(np.float32)

                G_err, _ = self.session.run([self.gen_cost_, self.gen_train_op],
                                            feed_dict={self.real_data: batch_SRVF, self.emotion: batch_label_emo,
                                                       self.first_frames_real: batch_first_frame, self.Joints_pos_real: batch_all_frames})
                for I in range(self.disc_iters):
                    D_err, _ = self.session.run([self.disc_cost, self.disc_train_op],
                                                feed_dict={self.real_data: batch_SRVF, self.emotion: batch_label_emo, self.first_frames_real: batch_first_frame})

                print("\nEpoch: [%3d/%3d] Batch: [%3d/%3d]\n\tD_err=%.4f \n\tG_err=%.4f" %
                      (epoch + 1, num_epochs, ind_batch + 1, num_batches, D_err, G_err))
                elapse = time.time() - start_time
                time_left = ((self.num_epochs - epoch - 1) * num_batches + (num_batches - ind_batch - 1)) * elapse
                print("\tTime left: %02d:%02d:%02d" %
                      (int(time_left / 3600), int(time_left % 3600 / 60), time_left % 60))
                summary = self.summary.eval(
                    feed_dict={
                        self.real_data: batch_SRVF,
                        self.emotion: batch_label_emo,
                        self.first_frames_real: batch_first_frame,
                        self.Joints_pos_real: batch_all_frames
                    }
                )

                self.writer.add_summary(summary, self.global_step.eval())
                if np.mod(epoch+1, 100) == 0:
                   self.save_checkpoint()
        self.save_checkpoint()
        self.writer.close()

    def Generator(self, y, first_frame, noise=None, reuse_variables=False, enable_tile_label=True, tile_ratio=1.0):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        if noise is None:
            noise = tf.random.normal([self.size_batch, 1])
            #fframe_to_concat = first_frame[:, :, 1]
            noise = tf.concat([noise, y], 1)
            #noise = y
        if (self.size_SRVF_H > self.size_SRVF_W):
            num_layers = int(np.log2(self.size_SRVF_W)) - int(self.size_kernel / 2)
        else:
            num_layers = int(np.log2(self.size_SRVF_H)) - int(self.size_kernel / 2)
        ## TODO: try concat_label used in ExprGAN. In this case, 6 channels will be added to the output without changing the size of feature maps
        duplicate = 1
        #z = concat_label_newtf(noise, y, duplicate=duplicate)
        z=noise
        size_mini_map_H = int(self.size_SRVF_H / 2 ** num_layers)
        size_mini_map_W = int(self.size_SRVF_W / 2 ** num_layers)
        name = 'G_fc'
        current = fc(
            input_vector=z,
            num_output_length=self.num_gen_channels * size_mini_map_H * size_mini_map_W,
            name=name
        )
        current = tf.reshape(current, [-1, size_mini_map_H, size_mini_map_W, self.num_gen_channels])
        current = tf.nn.relu(current)
        #current = concat_label_newtf(current, y) # remove intermediate concat
        for i in range(num_layers):
            name = 'G_deconv' + str(i)
            current = tf.compat.v1.image.resize_nearest_neighbor(current, [size_mini_map_H * 2 ** (i + 1),
                                                                           size_mini_map_W * 2 ** (i + 1)])
            current = custom_conv2d(input_map=current, num_output_channels=int(self.num_gen_channels / 2 ** (i + 1)),
                                    name=name)
            current = tf.nn.relu(current)
            #current = concat_label_newtf(current, y) # remove intermediate concat
        name = 'G_deconv' + str(i + 1)
        current = tf.compat.v1.image.resize_nearest_neighbor(current, [self.size_SRVF_H, self.size_SRVF_W])
        current = custom_conv2d(input_map=current, num_output_channels=int(self.num_gen_channels / 2 ** (i + 2)),
                                name=name)
        current = tf.nn.relu(current)

        #current = concat_label_newtf(current, y) #remove intermediate concat
        name = 'G_deconv' + str(i + 2)
        current = custom_conv2d(input_map=current, num_output_channels=self.num_input_channels,
                                name=name)  ### output format: NHWC
        generated_image_ = tf.nn.tanh(current)
        generated_image = tf.reshape(generated_image_, [self.size_batch, self.size_SRVF_H * self.size_SRVF_W])
        return generated_image

    def Discriminator(self, z, y, is_training=True, reuse_variables=False, num_hidden_layer_channels=(64, 32, 16),
                      enable_bn=True):
        if reuse_variables:
            tf.compat.v1.get_variable_scope().reuse_variables()
        num_layers = len(num_hidden_layer_channels)
        current = tf.reshape(z, [self.size_batch, self.size_SRVF_H, self.size_SRVF_W, 1])  ##generated_image
        current = concat_label_newtf(current, y)
        for i in range(num_layers):
            print(i)
            name = 'D_img_conv' + str(i)
            current = conv2d(
                input_map=current,
                num_output_channels=num_hidden_layer_channels[i],
                size_kernel=self.size_kernel,
                name=name
            )
            print(current.get_shape())
            if enable_bn:
                name = 'D_img_bn' + str(i)
                current = tf.compat.v1.layers.batch_normalization(
                    current,
                    scale=False,
                    training=is_training,
                    name=name,
                    reuse=reuse_variables
                )
            current = tf.nn.relu(current)
            #current = concat_label_newtf(current, y) #remove intermediate concat
            print(current.get_shape())
        name = 'D_img_fc1'
        current = fc(
            input_vector=tf.reshape(current, [self.size_batch, -1]),
            num_output_length=1024,
            name=name
        )
        name = 'D_img_fc1_bn'
        current = tf.compat.v1.layers.batch_normalization(
            current,
            scale=False,
            training=is_training,
            name=name,
            reuse=reuse_variables
        )
        current = lrelu(current)
        #current = concat_label_newtf(current, y) #remove intermediate concat
        name = 'D_img_fc2'
        disc = fc(
            input_vector=current,
            num_output_length=1,
            name=name
        )
        return disc

    def load_data(self, file):
        # get path to data file x = input, y= condition , frame = first frame of reaction sequence, gram = path to folder with gram matrix for each frame
        X = []
        y = []
        fframe = []
        frames =[]
        for line in open(file, 'r'):
            data = line.split()
            X.append(data[0])
            y.append(data[1])
            fframe.append(data[2])
            frames.append(data[3])
        seed = 2019
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)
        np.random.seed(seed)
        np.random.shuffle(fframe)
        np.random.seed(seed)
        np.random.shuffle(frames)
        return X, y, fframe,frames

    def load_data_test(self, file):
        # get path to data file x = input, y= condition , frame = first frame of reaction sequence, gram = path to folder with gram matrix for each frame
        X = []
        y = []
        fframe = []
        for line in open(file, 'r'):
            data = line.split()
            X.append(data[0])
            y.append(data[1])
            fframe.append(data[2])
        seed = 2019
        return X, y, fframe


    def read_SRVF(self, path_SRVF):
        data_ = loadmat(path_SRVF)
        data = data_['q2n']
        data = np.reshape(data, [data.shape[0] * data.shape[1]])
        return data

    def read_first_frame(self, path_frame):
        data_ = np.loadtxt(path_frame, delimiter=',')
        a = data_[:, 0]
        b = data_[:, 1]
        c = data_[:, 2]
        d = np.hstack([a, b])
        e = np.hstack([d, c])
        data = e
        for f in range(1, self.size_SRVF_W):
            data = np.vstack((data, e))
        data = np.transpose(data)
        return data

    def read_frames(self, path):
        data_ = loadmat(path)
        if "_A" in path:
            data = data_['curve_A']
        else:
            data = data_['curve_B']
        return data

    def Inner(self, A, B):
        [m, n, T] = A.get_shape().as_list()
        A_B = [A, B]
        mult = tf.map_fn(lambda a_b: a_b[0] * a_b[1], A_B, dtype=tf.float32)  ##A*A  ##tf.multiply(A,A)
        s1 = tf.reduce_sum(mult, 1, keepdims=False)
        s2 = tf.reduce_sum(s1, 1, keepdims=False) / T
        ##norm=tf.sqrt(s2)
        return s2

    def q_to_curves(self, q):

        s = tf.linspace(0.0, 1.0, self.size_SRVF_W)
        qnorme = tf.norm(q, ord=2, axis=[1, 3], keepdims=None, name=None)
        qnorm = tf.expand_dims(qnorme, 1)

        qnorm = tf.squeeze(qnorm)
        QN = tf.repeat(qnorm, repeats=self.size_SRVF_H, axis=0)
        QN = tf.reshape(QN, [self.size_batch, self.size_SRVF_H, self.size_SRVF_W])
        temp = tf.math.multiply(tf.squeeze(q), QN)
        curve = self.cumultrapz(temp, s)

        return curve

    def Gram_loss_func(self, Landmarks, Landmarks_real,first_frame):

        Landmarks = tf.reshape(Landmarks,[self.size_batch, self.size_SRVF_H, self.size_SRVF_W,1])
        if self.is_training:
            curves = self.q_to_curves(Landmarks)
            Lands = curves + first_frame
        else:
            Lands = tf.Variable(tf.zeros([self.size_batch, self.size_SRVF_H, self.size_SRVF_W], tf.float32))

        n_dim = 3
        Joints_real = tf.reshape(Landmarks_real, [self.size_batch, n_dim, int(self.size_SRVF_H / n_dim), self.size_SRVF_W])
        Joints_fake = tf.reshape(Lands,  [self.size_batch, n_dim, int(self.size_SRVF_H / n_dim), self.size_SRVF_W])
        G = tf.linalg.matmul(tf.transpose(Joints_fake, perm=[0, 3, 1, 2]), tf.transpose(Joints_real, perm=[0, 3, 2, 1]))
        sig, u, v = tf.linalg.svd(G)
        ssig = tf.reduce_sum(sig, axis=[2])
        Gram_Real = tf.linalg.matmul(tf.transpose(Joints_real, perm=[0, 3, 2, 1]), tf.transpose(Joints_real, perm=[0, 3, 1, 2]))
        Gram_Fake = tf.linalg.matmul(tf.transpose(Joints_fake, perm=[0, 3, 2, 1]), tf.transpose(Joints_fake, perm=[0, 3, 1, 2]))
        L = tf.linalg.trace(Gram_Real) + tf.linalg.trace(Gram_Fake) - 2.0 * ssig
        Gram_Loss = tf.reduce_mean(L)

        #Bones_Real = tf.gather_nd(Gram_Real, self.indices)
        #Bones_Real = tf.reshape(Bones_Real, [self.size_batch, self.size_SRVF_W, self.lenlinks])
        #Bones_Fake = tf.gather_nd(Gram_Fake, self.indices)
        #Bones_Fake = tf.reshape(Bones_Fake, [self.size_batch, self.size_SRVF_W, self.lenlinks])
        #diff= Bones_Real-Bones_Fake
        #norme = tf.norm(diff, ord='euclidean', axis=2)
        #Bone_Loss = tf.reduce_mean(norme)

        Joints_fake = tf.transpose(Joints_fake, perm=[0, 3, 1, 2])
        Joints_real = tf.transpose(Joints_real, perm=[0, 3, 1, 2])
        J1 = tf.gather_nd(Joints_real, self.indices_L)
        J1 = tf.reshape(J1, [self.size_batch, self.size_SRVF_W, 3, self.lenlinks])
        J2 = tf.gather_nd(Joints_real, self.indices_R)
        J2 = tf.reshape(J2, [self.size_batch, self.size_SRVF_W, 3, self.lenlinks])
        B1 = J1 - J2
        B1 = tf.norm(B1, ord='euclidean', axis=2)
        J3 = tf.gather_nd(Joints_fake, self.indices_L)
        J3 = tf.reshape(J3, [self.size_batch, self.size_SRVF_W, 3, self.lenlinks])
        J4 = tf.gather_nd(Joints_fake, self.indices_R)
        J4 = tf.reshape(J4, [self.size_batch, self.size_SRVF_W, 3, self.lenlinks])
        B2 = J3 - J4
        B2 = tf.norm(B2, ord='euclidean', axis=2)
        diff3 = B1 - B2
        norme_3 = tf.norm(diff3, ord='euclidean', axis=2)
        Bone_Loss = tf.reduce_mean(norme_3)

        return Gram_Loss,Bone_Loss

    def cumultrapz(self,y,x):
        dx = (x[1] - x[0])
        Y1 = y[:, :, 0:-1]
        Y2 = y[:, :, 1:]
        inte = ((Y1 + Y2) / 2.0) * dx
        zer = tf.Variable(tf.zeros([self.size_batch, self.size_SRVF_H, self.size_SRVF_W - 1], tf.float32))
        integ = tf.concat([zer, inte], axis=2)
        cumul = tf.math.cumsum(integ[:, :, self.size_SRVF_W - 2:], axis=2)
        return cumul

    def exp_map(self, q):
        q = tf.reshape(q, [self.size_batch, self.size_SRVF_H, self.size_SRVF_W])
        [m, n, T] = q.get_shape().as_list()
        lw = tf.sqrt(self.Inner(q, q))
        res = self.Q_ref_tensor * tf.expand_dims(tf.expand_dims(tf.cos(lw), -1), -1) + q * (
            tf.expand_dims(tf.expand_dims(tf.sin(lw) / lw, -1), -1))
        return tf.reshape(res, [self.size_batch, self.size_SRVF_H * self.size_SRVF_W])

    def log_map(self, q):
        q = tf.reshape(q, [self.size_batch, self.size_SRVF_H, self.size_SRVF_W])
        [m, n, T] = q.get_shape().as_list()
        prod = self.Inner(self.Q_ref_tensor, q)
        u = q - self.Q_ref_tensor * tf.expand_dims(tf.expand_dims(prod, -1), -1)
        u = tf.cast(u, tf.float32)
        lu = tf.sqrt(self.Inner(u, u))
        theta = tf.acos(tf.clip_by_value(prod, -0.98, 0.98))  ###tf.acos(prod)
        zero = tf.constant(0, shape=[m], dtype=tf.float32)

        def f1(): return tf.cast(u * tf.expand_dims(tf.expand_dims(zero, -1), -1), tf.float32)

        def f2(): return tf.cast(u * tf.expand_dims(tf.expand_dims(theta / lu, -1), -1), tf.float32)

        res = tf.cond(tf.reduce_all(tf.equal(lu, zero)), f1, f2)
        return tf.reshape(res, [self.size_batch, self.size_SRVF_H * self.size_SRVF_W])

    def geodesic_dist(self, q1, q2):
        q1 = tf.reshape(q1, [self.size_batch, self.size_SRVF_H, self.size_SRVF_W])
        q2 = tf.reshape(q2, [self.size_batch, self.size_SRVF_H, self.size_SRVF_W])
        inner_prod = self.Inner(q1, q2)
        dist = tf.acos(tf.clip_by_value(inner_prod, -0.98, 0.98))  ## ds=acos(<q1,q2>)
        return dist

    def save_checkpoint(self):
        checkpoint_dir = os.path.join(self.save_dir, self.checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(
            sess=self.session,
            save_path=os.path.join(checkpoint_dir, 'model'),
            global_step=self.global_step.eval()
        )

    def load_checkpoint(self, dir):
        print("\n\tLoading pre-trained model ...")
        checkpoint_dir = dir
        print(checkpoint_dir)
        checkpoints = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoints and checkpoints.model_checkpoint_path:
            checkpoints_name = self.checkpoint_name ##os.path.basename(checkpoints.model_checkpoint_path)
            self.saver.restore(self.session, os.path.join(checkpoint_dir, checkpoints_name))
            return True
        else:
            return False

    def y_to_rb_label(self, label):
        # inutilise
        number = np.argmax(label)
        one_hot = np.random.uniform(-1, 1, self.rb_dim)
        rb = np.tile(-1 * np.abs(one_hot), self.y_dim)
        rb[number * self.rb_dim:(number + 1) * self.rb_dim] = np.abs(one_hot)
        return rb

    def test(self, random_seed, dir):
        if not self.load_checkpoint(dir):
            print("\tFAILED >_<!")
            exit(0)
        else:
            print("\tSUCCESS ^_^")
        data_xtest, data_ytest, first_frame = self.load_data_test(self.load_test)
        data_xtest = [os.path.join(self.data_dir, x) for x in data_xtest]
        data_ytest = [os.path.join(self.data_dir, y) for y in data_ytest]
        first_frame = [os.path.join(self.data_dir, f) for f in first_frame]


        batch = [self.read_SRVF(path_SRVF=batch_file) for batch_file in data_xtest]
        batch_ff = [self.read_first_frame(path_frame=batch_file) for batch_file in first_frame]

        batch_label_rb = np.array(batch).astype(np.float32)
        batch_first_frame = np.array(batch_ff).astype(np.float32)

        SRVF_generated = tf.reshape(self.exp_fake, [len(data_xtest), self.size_SRVF_H, self.size_SRVF_W])
        norm = tf.sqrt(self.Inner(SRVF_generated, SRVF_generated))
        SRVF_generat = self.session.run(SRVF_generated, feed_dict={self.emotion: batch_label_rb,
                                                                   self.first_frames_real: batch_first_frame})

        res = self.session.run(norm,
                               feed_dict={self.emotion: batch_label_rb, self.first_frames_real: batch_first_frame})
        #mean_res = sum(res) /len(data_xtest)
        k = 0
        for i in range(0,len(res)):
            if 0.9 < res[i] < 1.1:
                k = k + 1

        batch_labels = [self.read_SRVF(path_SRVF=batch_file) for batch_file in data_ytest]
        for i in range(0,len(data_xtest)):
            nb = i + 1
            save_path = self.generated_dir + '/test_' + str(nb) + '.mat'
            scipy.io.savemat(save_path, dict([('x_test', SRVF_generat[i])]))

        print('Samples generated')

    def test_recursive(self,nb_recur, random_seed, dir):
        if not self.load_checkpoint(dir):
            print("\tFAILED >_<!")
            exit(0)
        else:
            print("\tSUCCESS ^_^")
        data_xtest, data_ytest, first_frame = self.load_data_test(self.load_test)
        data_xtest = [os.path.join(self.data_dir, x) for x in data_xtest]
        data_ytest = [os.path.join(self.data_dir, y) for y in data_ytest]
        first_frame = [os.path.join(self.data_dir, f) for f in first_frame]


        batch = [self.read_SRVF(path_SRVF=batch_file) for batch_file in data_xtest]
        batch_ff = [self.read_first_frame(path_frame=batch_file) for batch_file in first_frame]

        for r in range(nb_recur):
            if r>0:
                batch= np.reshape(SRVF_generat[:,:,1:], (self.size_batch, self.size_SRVF_H * (self.size_SRVF_W-1)))
            batch_label_rb = np.array(batch).astype(np.float32)
            batch_first_frame = np.array(batch_ff).astype(np.float32)

            SRVF_generated = tf.reshape(self.exp_fake, [len(data_xtest), self.size_SRVF_H, self.size_SRVF_W])
            norm = tf.sqrt(self.Inner(SRVF_generated, SRVF_generated))
            SRVF_generat = self.session.run(SRVF_generated, feed_dict={self.emotion: batch_label_rb,
                                                                       self.first_frames_real: batch_first_frame})

            res = self.session.run(norm,
                                   feed_dict={self.emotion: batch_label_rb, self.first_frames_real: batch_first_frame})
            k = 0
            for i in range(0,len(res)):
                if 0.9 < res[i] < 1.1:
                    k = k + 1
            if not os.path.exists('samples_recursive/recur_'+ str(r)):
                os.makedirs('samples_recursive/recur_'+ str(r))
            for i in range(0,len(data_xtest)):
                nb = i + 1
                save_path = 'samples_recursive/recur_'+ str(r)+'/test_' + str(nb) + '.mat'
                scipy.io.savemat(save_path, dict([('x_test', SRVF_generat[i])]))

        print('Samples generated')


