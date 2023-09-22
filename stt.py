from numpy import frombuffer, int16
from pyaudio import PyAudio,paInt16
import whisper
from warnings import filterwarnings

filterwarnings('ignore')

# Set up PyAudio
chunk = 1024  # Record in chunks of 1024 samples
sample_format = paInt16  # 16 bits per sample
channels = 1
fs = 16000  # Record at 16kHz
seconds = 4
p = PyAudio()
model = whisper.load_model("small.en")


def transcribe_audio():
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames
    print("Recording...")
    # Store data in chunks for 'seconds' number of seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream 
    stream.stop_stream()
    stream.close()
    print("Recording completed.")
    
    # Convert the byte data to a numpy array
    audio_data = frombuffer(b''.join(frames), dtype=int16)
    # Convert the audio data to float32 format
    audio_data = audio_data.astype('float32') / 32767.0
    # Transcribe the audio
    result = model.transcribe(audio_data)
    print(result["text"])
    # Terminate the PortAudio interface
    p.terminate()
transcribe_audio()