import wave
with wave.open('/home/bao/Documents/AI/AI_data_train/aoquan/aoquan_0001.wav', 'rb') as wav_file:
    sample_rate = wav_file.getframerate()
    print(f"Tần số lấy mẫu: {sample_rate} Hz")


# import os
# import librosa
# import soundfile as sf
#
# # Đường dẫn gốc
# root_dir = "/home/bao/Documents/AI/AI_data_train"
#
# # Duyệt qua tất cả các thư mục con
# for folder_name in os.listdir(root_dir):
#     folder_path = os.path.join(root_dir, folder_name)
#     if os.path.isdir(folder_path):
#         for file_name in os.listdir(folder_path):
#             if file_name.endswith(".wav"):
#                 file_path = os.path.join(folder_path, file_name)
#                 # Đọc file WAV với tần số gốc 48000 Hz
#                 audio, sr = librosa.load(file_path, sr=48000)
#                 # Chuyển đổi tần số xuống 16000 Hz
#                 audio_resampled = librosa.resample(audio, orig_sr=48000, target_sr=16000)
#                 # Lưu file mới, ghi đè lên file gốc
#                 sf.write(file_path, audio_resampled, 16000)
#                 print(f"Đã chuyển đổi: {file_path}")