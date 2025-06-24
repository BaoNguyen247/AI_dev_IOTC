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
tf.config.optimizer.set_experimental_options({"layout_optimizer": False})
from mltk.core.preprocess.audio.audio_feature_generator import AudioFeatureGeneratorSettings
class MyModel(
    MltkModel,
    TrainMixin,
    ImageDatasetMixin,
    EvaluateClassifierMixin
):
    pass
my_model = MyModel()

# General parameters
my_model.version = 2
my_model.description = 'TinyML: Keyword spotting for Vietnamese words - dsconv_arm'

#################################################
# Training parameters
my_model.epochs = 80
my_model.batch_size = 32
my_model.optimizer = 'adam'
my_model.metrics = ['accuracy']
my_model.loss = 'categorical_crossentropy'

#################################################
# Image Dataset Settings

# The directory of the training data
def prepare_local_dataset():
    """Use local dataset from /home/bao/Documents/AI/AI_data_train_spectrograms/"""
    return '/content/drive/MyDrive/AI_data_storage/spectrogram'


frontend_settings = AudioFeatureGeneratorSettings()

frontend_settings.sample_rate_hz = 16000
frontend_settings.sample_length_ms = 1000
frontend_settings.window_size_ms = 32
frontend_settings.window_step_ms = 16
frontend_settings.filterbank_n_channels = 64
frontend_settings.filterbank_upper_band_limit = 4000.0-1 # Spoken language usually only goes up to 4k
frontend_settings.filterbank_lower_band_limit = 100.0
frontend_settings.noise_reduction_enable = False # Disable the noise reduction block
frontend_settings.noise_reduction_smoothing_bits = 5
frontend_settings.noise_reduction_even_smoothing = 0.004
frontend_settings.noise_reduction_odd_smoothing = 0.004
frontend_settings.noise_reduction_min_signal_remaining = 0.05
frontend_settings.pcan_enable = False
frontend_settings.pcan_strength = 0.95
frontend_settings.pcan_offset = 80.0
frontend_settings.pcan_gain_bits = 21
frontend_settings.log_scale_enable = True
frontend_settings.log_scale_shift = 6

frontend_settings.activity_detection_enable = True # Enable the activity detection block
frontend_settings.activity_detection_alpha_a = 0.5
frontend_settings.activity_detection_alpha_b = 0.8
frontend_settings.activity_detection_arm_threshold = 0.75
frontend_settings.activity_detection_trip_threshold = 0.8

frontend_settings.dc_notch_filter_enable = True # Enable the DC notch filter
frontend_settings.dc_notch_filter_coefficient = 0.95

frontend_settings.quantize_dynamic_scale_enable = True # Enable dynamic quantization
frontend_settings.quantize_dynamic_scale_range_db = 40.0


# See https://siliconlabs.github.io/mltk/docs/guides/model_parameters.html
my_model.model_parameters.update(frontend_settings)
my_model.dataset = prepare_local_dataset
my_model.class_mode = 'categorical'
my_model.classes = ('aoquan', 'ca', 'caphe', 'chebiensan', 'daydep', 'gao', 'giaikhat', 'giavi', 'giay', 'haisan', 'hop', 'lanh', 'nuocuong', 'raucu', 'sachvo', 'sua', 'thit', 'tra', 'traicay', 'unknow', 'vat')
my_model.class_weights = 'balanced' # Ensure the classes samples a balanced during training
my_model.input_shape = frontend_settings.spectrogram_shape + (1,)
my_model.tflite_converter['optimizations'] = [tf.lite.Optimize.DEFAULT]
my_model.tflite_converter['supported_ops'] = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
my_model.tflite_converter['inference_input_type'] = np.int8
my_model.tflite_converter['inference_output_type'] = np.int8
 # generate a representative dataset from the validation data
my_model.tflite_converter['representative_dataset'] = 'generate'
validation_split = 0.1
my_model.model_parameters['average_window_duration_ms'] = 1000
my_model.model_parameters['detection_threshold'] = 165
my_model.model_parameters['suppression_ms'] = 750
my_model.model_parameters['minimum_count'] = 3
my_model.model_parameters['volume_gain'] = 2
my_model.model_parameters['latency_ms'] = 100
my_model.model_parameters['verbose_model_output_logs'] = False

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

my_model.reduce_lr_on_plateau = dict(
  monitor='accuracy',
  factor = 0.95,
  patience = 1,
  min_delta=0.01
)

# https://keras.io/api/callbacks/early_stopping/
# If the validation accuracy doesn't improve after 'patience' epochs then stop training
my_model.early_stopping = dict(
  monitor = 'val_accuracy',
  patience = 15
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
my_model.tensorboard = dict(
    histogram_freq=0,       # frequency (in epochs) at which to compute activation and weight histograms
                            # for the layers of the model. If set to 0, histograms won't be computed.
                            # Validation data (or split) must be specified for histogram visualizations.
    write_graph=False,       # whether to visualize the graph in TensorBoard. The log file can become quite large when write_graph is set to True.
    write_images=False,     # whether to write model weights to visualize as image in TensorBoard.
    update_freq="batch",    # 'batch' or 'epoch' or integer. When using 'batch', writes the losses and metrics
                            # to TensorBoard after each batch. The same applies for 'epoch'.
                            # If using an integer, let's say 1000, the callback will write the metrics and losses
                            # to TensorBoard every 1000 batches. Note that writing too frequently to
                            # TensorBoard can slow down your training.
    profile_batch=(51,51),        # Profile the batch(es) to sample compute characteristics.
                            # profile_batch must be a non-negative integer or a tuple of integers.
                            # A pair of positive integers signify a range of batches to profile.
                            # By default, it will profile the second batch. Set profile_batch=0 to disable profiling.
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
    #convert_wav_to_spectrogram('/content/drive/MyDrive/AI_data_train', '/content/drive/MyDrive/AI_data_storage/spectrogram')

    test_mode_enabled = False

    train_results = mltk_core.train_model(my_model, clean=True, test=test_mode_enabled)
    print(train_results)

    tflite_eval_results = mltk_core.evaluate_model(my_model, verbose=True, test=test_mode_enabled)
    print(tflite_eval_results)

    profiling_results = mltk_core.profile_model(my_model, test=test_mode_enabled)
    print(profiling_results)