import tensorflow as tf

from . import vggish_postprocess
from . import vggish_input
from . import vggish_params
from . import vggish_slim


def run_inference_vgg(input_file):
    with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, "audioset/vggish_model.ckpt")
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)

        examples_batch = vggish_input.wavfile_to_examples(input_file)

        if examples_batch.shape[0] == 0:
            print("File couldn't be processed")
            return None
        else:
            [embedding_batch] = sess.run([embedding_tensor],
                                         feed_dict={features_tensor: examples_batch})
            postprocessed_batch = vggish_postprocess.Postprocessor(
                "audioset/vggish_pca_params.npz").postprocess(embedding_batch)
            return postprocessed_batch
