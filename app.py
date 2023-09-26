from flask import Flask, render_template, request
import pyaudio
import wave
import scipy.io.wavfile as wav
import noisereduce as nr
import numpy as np
import librosa
import joblib
from textblob import TextBlob
import scipy.stats as stats
from scipy.stats import entropy



app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/index', methods = ['GET', 'POST'])
def index():
    prediction = None
    intensity = None
    if request.method == 'POST':
        name = request.form['name']
        gender = request.form['gender']

        #Audio input
        path = f'D:/SPEECH EMOTION RECOGNITION/{name}_{gender}.wav'
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 1024
        RECORD_SECONDS = 11

        # Initialize PyAudio
        audio = pyaudio.PyAudio()

        # Open the audio stream
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

        print("Recording started...")
        frames = []
        # Record audio data
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("Recording finished.")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        wave_file = wave.open(path, 'wb')
        wave_file.setnchannels(CHANNELS)
        wave_file.setsampwidth(audio.get_sample_size(FORMAT))
        wave_file.setframerate(RATE)
        wave_file.writeframes(b''.join(frames))
        wave_file.close()
        print("Audio saved to", path)
        sample_rate, data = wav.read(f'D:/SPEECH EMOTION RECOGNITION/{name}_{gender}.wav')


        # Apply noise reduction
        if len(data.shape) > 1:
            data = data[:, 0]
        reduced_noise = nr.reduce_noise(y=data, sr=sample_rate)
        wav.write(f'D:/SPEECH EMOTION RECOGNITION/{name}_{gender}_nr.wav', sample_rate, reduced_noise.astype(np.int16))

        audio, sr = librosa.load(path)
        mean_frequency = librosa.feature.spectral_centroid(y=audio, sr=sr)
        maximum = np.max(mean_frequency)
        minimum = np.min(mean_frequency)

        meanfreq = (np.mean(mean_frequency) - minimum) / (maximum - minimum)

        normalized_audio = (audio - np.min(audio)) / (np.max(audio) - np.min(audio))
        sd = np.std(normalized_audio)

        sorted_audio = sorted(audio)
        num_samples = len(sorted_audio)
        if num_samples % 2 == 1:
            median_nn = sorted_audio[num_samples // 2]
        else:
            middle_left = sorted_audio[num_samples // 2 - 1]
            middle_right = sorted_audio[num_samples // 2]
            median_nn = (middle_left + middle_right) / 2.0

        # Normalize the median to the range [0, 1]
        min_val = min(audio)
        max_val = max(audio)
        median = (median_nn - min_val) / (max_val - min_val)

        quartiles = np.percentile(audio, [0, 25, 75, 100])
        q25 = (quartiles[1] - quartiles[0]) / (quartiles[3] - quartiles[0])
        q75 = (quartiles[2] - quartiles[0]) / (quartiles[3] - quartiles[0])
        iqr = q75 - q25

        skew = stats.skew(audio)

        kurt = stats.kurtosis(audio)

        specgram = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
        spectral_energy = np.sum(specgram, axis=0)
        spectral_energy /= np.sum(spectral_energy)
        sp_ent = entropy(spectral_energy) / 10

        n_fft = 2048
        hop_length = 512
        stft_result = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude_spectrum = np.abs(stft_result)
        sfm = np.exp(np.mean(np.log(np.maximum(1e-10, magnitude_spectrum))) - np.mean(np.log(np.maximum(1e-10, np.mean(magnitude_spectrum)))))

        mode = np.abs(stats.mode(audio, keepdims = True))[0][0]

        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
        centroid_mean = np.mean(spectral_centroids)
        centroid_max = np.max(spectral_centroids)
        centroid_min = np.min(spectral_centroids)
        centroid = (centroid_mean - centroid_min) / (centroid_max - centroid_min)

        pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sr)
        pitches = pitches[pitches > 0]
        meanfun = np.mean(pitches)
        maxfun = np.max(pitches)
        minfun = np.min(pitches)
        meanfun = (meanfun - minfun) / (maxfun - minfun)
        minfun = minfun / maxfun
        normalized_pitches = (pitches - minfun) / (maxfun - minfun)
        maxfun = np.mean(normalized_pitches)
        minfun = minfun / maxfun

        import speech_recognition as sr
        # Initialize the recognizer
        recognizer = sr.Recognizer()
        # Load the audio file
        with sr.AudioFile(path) as source:
            audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            text = "Speech Recognition could not understand the audio."
        except sr.RequestError as e:
            text = "Could not request results from Speech Recognition service."

        # Emotional intensity
        def calculate_emotional_intensity(text):
            blob = TextBlob(text)
            sentiment_score = blob.sentiment.polarity
            normalized_intensity = min(1, max(-1, sentiment_score))
            return normalized_intensity

        emotional_intensity = calculate_emotional_intensity(text)

        # Prediction
        if(gender == 'Female'):
            with open('svm_female_joblib.joblib', 'rb') as file:
                loaded_model = joblib.load(file)
                lst = [meanfreq, sd, median, q25, q75, iqr, skew, kurt, sp_ent, sfm, mode, centroid, meanfun]
                output = loaded_model.predict([lst])
        else:
            with open('svm_male_joblib.joblib','rb') as file:
                loaded_model = joblib.load(file)
                lst = [meanfreq, sd, median, q25, q75, iqr, skew, kurt, sp_ent, sfm, mode, centroid, meanfun, minfun, maxfun]
                output = loaded_model.predict([lst])
        output = list(output)[0]

        if(emotional_intensity == 0):
            prediction = 'Calm'
        elif(emotional_intensity < 0 and output == 2):
            prediction = "Angry"
        else:
            if(output == 0):
                prediction = 'Angry'
            elif(output == 1):
                prediction = 'Fear'
            elif(output == 2):
                prediction = 'Happy'
            elif(output == 3):
                prediction = 'Sad'
            else:
                prediction = 'Calm'

        emotional_intensity = round(emotional_intensity, 3)
        if(emotional_intensity == 0):
            intensity = str(emotional_intensity) + '(Normal)'
        elif(emotional_intensity <= 0.5 and emotional_intensity >= -0.5):
            intensity = str(emotional_intensity) + '  (Weak)'
        else:
            intensity = str(emotional_intensity) +'  (Strong)' 

    if(prediction == None and intensity == None):
        vals = ["None","None"]
    else:
        vals = [prediction, intensity]
    
    return render_template('index.html', vals = vals)
if __name__ == '__main__':
    app.run(debug = True)