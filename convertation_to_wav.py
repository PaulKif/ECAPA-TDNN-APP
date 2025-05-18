import librosa
import soundfile as sf
import os

def convert_mp3_to_wav(mp3_folder, wav_folder):
    if not os.path.exists(wav_folder):
        os.makedirs(wav_folder)
    
    for file_name in os.listdir(mp3_folder):
        if file_name.endswith('.mp3'):
            mp3_path = os.path.join(mp3_folder, file_name)
            wav_path = os.path.join(wav_folder, file_name.replace('.mp3', '.wav'))
            
            # Load MP3 file
            y, sr = librosa.load(mp3_path, sr=None)
            # Save as WAV
            sf.write(wav_path, y, sr)

convert_mp3_to_wav('ru/clips', 'ru/wav')
