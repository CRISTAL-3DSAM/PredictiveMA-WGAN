import tensorflow as tf
import time
from SRVF_WGAN_human import SRVF_WGAN


Training = False #choose between training and testing
Short_Term = False # choose between short term prediction and long term prediction


flags = tf.compat.v1.flags
flags.DEFINE_integer(flag_name='epoch', default_value=500, docstring='number of epochs')
flags.DEFINE_integer(flag_name='nb_joints', default_value=17, docstring='number of joint in the skeletons')
flags.DEFINE_integer(flag_name='Coeff_Gram', default_value=0, docstring='coeff of gram loss')
flags.DEFINE_integer(flag_name='Coeff_bone', default_value=10, docstring='coeff of bone loss')
if Short_Term:
    flags.DEFINE_integer(flag_name='number_frames', default_value=11, docstring='number of frames to generate')
    flags.DEFINE_integer(flag_name='number_frames_prior', default_value=10, docstring='number of frames to use as prior')
    flags.DEFINE_string(flag_name='data_dir', default_value='Data_skeleton', docstring='name the directory for the data') # can be different for long term
    flags.DEFINE_string(flag_name='load_train', default_value='Data_skeleton/Data_train_short.txt', docstring='train file to load')
    flags.DEFINE_string(flag_name='load_test', default_value='Data_skeleton/Data_test_short.txt', docstring='test file to load')
    flags.DEFINE_string(flag_name='load_qmean', default_value='Data_skeleton/q_mean_data_short.mat', docstring='qmean to load')
    flags.DEFINE_string(flag_name='save_dir', default_value='save_short', docstring='dir for saving training results')
    flags.DEFINE_string(flag_name='checkpoint_dir', default_value='Checkpoint_short', docstring='dir for loading checkpoints')
    flags.DEFINE_string(flag_name='checkpoint_name', default_value='model-135000', docstring='name of checkpoint file, model-XXXXX')
    flags.DEFINE_string(flag_name='generated_dir', default_value='generated_samples_short', docstring='name the directory for generated samples')
    if Training:
        flags.DEFINE_boolean(flag_name='is_train', default_value=True, docstring='training mode')
        flags.DEFINE_integer(flag_name='batch_size', default_value=64, docstring='number of batch size')

    else:
        flags.DEFINE_boolean(flag_name='is_train', default_value=False, docstring='training mode')
        flags.DEFINE_integer(flag_name='batch_size', default_value=812, docstring='number of samples in test set')

else:
    flags.DEFINE_integer(flag_name='number_frames', default_value=26, docstring='number of frames to generate')
    flags.DEFINE_integer(flag_name='number_frames_prior', default_value=25, docstring='number of frames to use as prior')
    flags.DEFINE_string(flag_name='data_dir', default_value='Data_skeleton', docstring='name the directory for the data')
    flags.DEFINE_string(flag_name='load_train', default_value='Data_skeleton/Data_train_long.txt', docstring='train file to load')
    flags.DEFINE_string(flag_name='load_test', default_value='Data_skeleton/Data_test_long.txt', docstring='test file to load')
    flags.DEFINE_string(flag_name='load_qmean', default_value='Data_skeleton/q_mean_data_long.mat', docstring='qmean to load')
    flags.DEFINE_string(flag_name='save_dir', default_value='save_long', docstring='dir for saving training results')
    flags.DEFINE_string(flag_name='checkpoint_dir', default_value='Checkpoint_long', docstring='dir for loading checkpoints')
    flags.DEFINE_string(flag_name='checkpoint_name', default_value='model-107500', docstring='name of checkpoint file, model-XXXXX')
    flags.DEFINE_string(flag_name='generated_dir', default_value='generated_samples_long', docstring='name the directory for generated samples')
    if Training:
        flags.DEFINE_boolean(flag_name='is_train', default_value=True, docstring='training mode')
        flags.DEFINE_integer(flag_name='batch_size', default_value=64, docstring='number of batch size')

    else:
        flags.DEFINE_boolean(flag_name='is_train', default_value=False, docstring='training mode')
        flags.DEFINE_integer(flag_name='batch_size', default_value=644, docstring='number of samples in test set')

FLAGS = flags.FLAGS

def main(_):
    import pprint
    #pprint.pprint(FLAGS.__flags)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    start_time = time.time()
    with tf.compat.v1.Session(config=config) as session:
        model = SRVF_WGAN(
            session,
            is_training=FLAGS.is_train,
            save_dir=FLAGS.save_dir,
            checkpoint_dir=FLAGS.checkpoint_dir,
            size_batch=FLAGS.batch_size,
            load_train=FLAGS.load_train,
            load_test = FLAGS.load_test,
            size_SRVF_W = FLAGS.number_frames,
            nb_frames = FLAGS.number_frames_prior,
            size_SRVF_H = FLAGS.nb_joints*3,
            coeff_skel_loss=FLAGS.Coeff_Gram,
            Bone_Loss_coeff =FLAGS.Coeff_bone,
            load_qmean = FLAGS.load_qmean,
            checkpoint_name=FLAGS.checkpoint_name,
            generated_dir = FLAGS.generated_dir,
            data_dir = FLAGS.data_dir

        )
        if FLAGS.is_train:
            print('\n\tTraining Mode')
            model.train(
                num_epochs=FLAGS.epoch  # reduit pour rapidite
            )
        else:
            seed = 2019
            print('\n\tTesting Mode')
            model.test_recursive(10,random_seed=seed, dir=FLAGS.checkpoint_dir,
                                 # label 1,2,3 pour premier SRVF, label 4,5,6 pour deuxieme SRVF
                                 )
    print('done')
    print("--- %s seconds ---" % round(time.time() - start_time, 2))

if __name__ == '__main__':
    tf.compat.v1.app.run()

