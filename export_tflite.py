import glob
import os
import argparse
import random
from tqdm import tqdm
import tensorflow as tf


def tf2tflite(input_model_dir_path, output_model_file_path, representative_dataset_gen, input_height, input_width):
    os.makedirs(os.path.dirname(output_model_file_path), exist_ok=True)

    loaded = tf.saved_model.load(input_model_dir_path)
    concrete_func = loaded.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    concrete_func.inputs[0].set_shape([1, input_height, input_width, 3])
    concrete_func.outputs[0].set_shape([1, input_width // 16, concrete_func.outputs[0].shape[-1]])
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.int8
    converter.representative_dataset = representative_dataset_gen

    tflite_quant_model = converter.convert()
    open(output_model_file_path, "wb").write(tflite_quant_model)


def main(input_model_dir_path, train_input_dir_path, output_model_file_path, input_height, input_width, sample_max_num):
    def representative_dataset_gen():
        for step_index, image_path in tqdm(enumerate(image_path_list), desc=f'sample_max_num:{sample_max_num}'):
            if step_index > sample_max_num:
                break
            yield [
                tf.cast(tf.expand_dims(tf.image.decode_image(tf.io.read_file(image_path), channels=3), 0), tf.float32)]

    types = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    image_path_list = []
    for files in types:
        image_path_list.extend(glob.glob(os.path.join(train_input_dir_path, files), recursive=True))
    random.shuffle(image_path_list)

    tf2tflite(input_model_dir_path, output_model_file_path, representative_dataset_gen, input_height, input_width)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='export')
    parser.add_argument('--input_model_dir_path', type=str,
                        default='~/.vaik_text_recognition_pb_trainer/output_model/2023-02-10-07-10-32/step-5000_batch-16_epoch-113_loss_0.1720_val_loss_0.8170_train_20h_exscore_0.88_tcn',
                        help="input tensor model dir path")
    parser.add_argument('--train_input_dir_path', type=str,
                        default='~/.vaik_text_recognition_pb_trainer/dump_dataset')
    parser.add_argument('--output_model_file_path', type=str,
                        default='~/.vaik_text_recognition_pb_exporter/model.tflite',
                        help="output tflite model file path")
    parser.add_argument('--input_height', type=int, default=96)
    parser.add_argument('--input_width', type=int, default=576)
    parser.add_argument('--sample_max_num', type=int, default=12500, help="output tflite model dir path")
    args = parser.parse_args()

    args.input_model_dir_path = os.path.expanduser(args.input_model_dir_path)
    args.train_input_dir_path = os.path.expanduser(args.train_input_dir_path)
    args.output_model_file_path = os.path.expanduser(args.output_model_file_path)

    main(args.input_model_dir_path, args.train_input_dir_path, args.output_model_file_path, args.input_height, args.input_width, args.sample_max_num)
