import tensorflow as tf

from .vggish_params import INPUT_TENSOR_NAME, OUTPUT_TENSOR_NAME
from .vggish_postprocess import Postprocessor
from .vggish_input import wavfile_to_examples
from .vggish_slim import define_vggish_slim, load_vggish_slim_checkpoint

sess = tf.Session()
define_vggish_slim(training=False)
load_vggish_slim_checkpoint(sess, "audioset/vggish_model.ckpt")
features_tensor = sess.graph.get_tensor_by_name(
    INPUT_TENSOR_NAME)
embedding_tensor = sess.graph.get_tensor_by_name(
    OUTPUT_TENSOR_NAME)


def run_inference_vgg(input_file):
    # Define the model in inference mode, load the checkpoint, and
    # locate input and output tensors.

    examples_batch = wavfile_to_examples(input_file)

    if examples_batch.shape[0] == 0:
        print("File couldn't be processed")
        return None
    else:
        [embedding_batch] = sess.run([embedding_tensor],
                                     feed_dict={features_tensor: examples_batch})
        postprocessed_batch = Postprocessor(
            "audioset/vggish_pca_params.npz").postprocess(embedding_batch)
        return postprocessed_batch
