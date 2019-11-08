from clone_tts import CloneTTS
import IPython.display as ipd
from generate_corpus import float2pcm

def test_clone():
    clone_tts = CloneTTS()
    clone_tts.load_models()
    # tts_audio = clone_tts.synthesize_clone('./nicole_sample.wav', "This is a test")
    nicole_tts = clone_tts.synthesizer_for('./nicole_sample.wav')
    tts_audio = nicole_tts("Oak is strong and also gives shade.")
    data_arr = tts_audio
    data = float2pcm(data_arr)
    ipd.Audio(float2pcm(data_arr),rate=clone_tts.synthesizer.sample_rate)
