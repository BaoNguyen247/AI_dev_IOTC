from mltk.core.preprocess.image.parallel_generator import ParallelImageDataGenerator
from mltk.core.model import (
    MltkModel,
    TrainMixin,
    ImageDatasetMixin,
    EvaluateClassifierMixin
)
from mltk.models.shared import DepthwiseSeparableConv2D_ARM
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
class MyModel(
    MltkModel,
    TrainMixin,
    ImageDatasetMixin,
    EvaluateClassifierMixin
):
    pass
my_model = MyModel()

# General parameters
my_model.version = 1
my_model.description = 'TinyML: Keyword spotting for Vietnamese words - dsconv_arm'

#################################################
# Training parameters
my_model.epochs = 80
my_model.batch_size = 64
my_model.optimizer = 'adam'
my_model.metrics = ['accuracy']
my_model.loss = 'categorical_crossentropy'

#################################################
# Image Dataset Settings

# The directory of the training data
def prepare_local_dataset():
    """Use local dataset from /home/bao/Documents/AI/AI_data_train_spectrograms/"""
    return '/home/bao/Documents/AI/AI_data_train_spectrograms'

my_model.dataset = prepare_local_dataset
# The classification type
my_model.class_mode = 'categorical'
# The class labels found in your training dataset directory
my_model.classes = ('aoquan', 'ca', 'caphe', 'chebiensan', 'daydep', 'gao', 'giaikhat', 'giavi', 'giay', 'haisan', 'hop', 'lanh', 'nuocuong', 'raucu', 'sachvo', 'sua', 'thit', 'tra', 'traicay', 'unknow', 'vat')
# The input shape to the model
my_model.input_shape = (50, 10, 1)
# These are the settings used to quantize the model
# We want all the internal ops as well as
# model input/output to be int8
my_model.tflite_converter['optimizations'] = [tf.lite.Optimize.DEFAULT]
my_model.tflite_converter['supported_ops'] = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
my_model.tflite_converter['inference_input_type'] = np.int8
my_model.tflite_converter['inference_output_type'] = np.int8
# Automatically generate a representative dataset from the validation data
my_model.tflite_converter['representative_dataset'] = 'generate'
validation_split = 0.1

##############################################################
# Training callbacks
#

def convert_wav_to_spectrogram(dataset_dir, output_dir):
    """Convert WAV files to spectrogram images and save them as JPG."""
    os.makedirs(output_dir, exist_ok=True)
    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_path):
            output_class_path = os.path.join(output_dir, class_name)
            os.makedirs(output_class_path, exist_ok=True)
            for wav_file in os.listdir(class_path):
                if wav_file.endswith('.wav'):
                    wav_path = os.path.join(class_path, wav_file)
                    y, sr = librosa.load(wav_path)
                    # Convert to spectrogram
                    spectrogram = np.abs(librosa.stft(y))
                    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
                    # Resize to match input_shape (50, 10)
                    from scipy.ndimage import zoom
                    target_shape = (50, 10)
                    spectrogram_resized = zoom(spectrogram_db, (target_shape[0] / spectrogram_db.shape[0], target_shape[1] / spectrogram_db.shape[1]))
                    # Ensure 2D for saving as grayscale
                    spectrogram_resized = np.clip(spectrogram_resized, -80, 0)  # Clip to typical dB range
                    # Save as JPG
                    output_file = os.path.join(output_class_path, f'{os.path.splitext(wav_file)[0]}.jpg')
                    plt.imsave(output_file, spectrogram_resized, cmap='gray', vmin=-80, vmax=0)
                    print(f"Saved spectrogram: {output_file}, shape: {spectrogram_resized.shape}")

# Learning rate schedule
def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 30:
        lrate = 0.0005
    if epoch > 40:
        lrate = 0.00025
    if epoch > 50:
        lrate = 0.00025
    if epoch > 60:
        lrate = 0.0001
    return lrate

my_model.lr_schedule = dict(
    schedule=lr_schedule,
    verbose=1
)

my_model.datagen = ParallelImageDataGenerator(
    cores=.35,
    max_batches_pending=32,
    rotation_range=0,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=(0.95, 1.05),
    validation_split=validation_split
)

##############################################################
# Model Layout
def my_model_builder(model: MyModel):
    num_classes = len(model.classes)  # Should be 21
    keras_model = Sequential([
        DepthwiseSeparableConv2D_ARM(input_shape=model.input_shape),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')  # Output layer with 21 units for 21 classes
    ])
    keras_model.compile(
        loss=model.loss,
        optimizer=model.optimizer,
        metrics=model.metrics
    )
    return keras_model

my_model.build_model_function = my_model_builder

##########################################################################################
# The following allows for running this model training script directly, e.g.:
# python keyword_spotting.py
#
if __name__ == '__main__':
    import mltk.core as mltk_core
    from mltk import cli

    cli.get_logger(verbose=False)

    # Convert WAV to spectrograms before training
    #convert_wav_to_spectrogram('/home/bao/Documents/AI/AI_data_train/', '/home/bao/Documents/AI/AI_data_train_spectrograms/')

    test_mode_enabled = False

    train_results = mltk_core.train_model(my_model, clean=True, test=test_mode_enabled)
    print(train_results)

    tflite_eval_results = mltk_core.evaluate_model(my_model, verbose=True, test=test_mode_enabled)
    print(tflite_eval_results)

    profiling_results = mltk_core.profile_model(my_model, test=test_mode_enabled)
    print(profiling_results)