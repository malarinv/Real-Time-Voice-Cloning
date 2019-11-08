from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import argparse
import torch
import sys


class CloneTTS(object):
    """docstring for CloneTTS."""

    def __init__(
        self,
        enc_model_fpath=Path("encoder/saved_models/pretrained.pt"),
        syn_model_dir=Path("synthesizer/saved_models/logs-pretrained/"),
        voc_model_fpath=Path("vocoder/saved_models/pretrained/pretrained.pt"),
        low_mem=False,
    ):
        super(CloneTTS, self).__init__()
        self.enc_model_fpath = enc_model_fpath
        self.syn_model_dir = syn_model_dir
        self.voc_model_fpath = voc_model_fpath
        self.low_mem = low_mem

    def load_models(self):
        print("Running a test of your configuration...\n")
        if not torch.cuda.is_available():
            print(
                "Your PyTorch installation is not configured to use CUDA. If you have a GPU ready "
                "for deep learning, ensure that the drivers are properly installed, and that your "
                "CUDA version matches your PyTorch installation. CPU-only inference is currently "
                "not supported.",
                file=sys.stderr,
            )
            quit(-1)
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        print(
            "Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
            "%.1fGb total memory.\n"
            % (
                torch.cuda.device_count(),
                device_id,
                gpu_properties.name,
                gpu_properties.major,
                gpu_properties.minor,
                gpu_properties.total_memory / 1e9,
            )
        )
        print("Preparing the encoder, the synthesizer and the vocoder...")
        encoder.load_model(self.enc_model_fpath)
        synthesizer = Synthesizer(
            self.syn_model_dir.joinpath("taco_pretrained"), low_mem=self.low_mem
        )
        vocoder.load_model(self.voc_model_fpath)
        print("Testing your configuration with small inputs.")
        print("\tTesting the encoder...")
        encoder.embed_utterance(np.zeros(encoder.sampling_rate))
        embed = np.random.rand(speaker_embedding_size)
        embed /= np.linalg.norm(embed)
        embeds = [embed, np.zeros(speaker_embedding_size)]
        texts = ["test 1", "test 2"]
        print(
            "\tTesting the synthesizer... (loading the model will output a lot of text)"
        )
        mels = synthesizer.synthesize_spectrograms(texts, embeds)
        mel = np.concatenate(mels, axis=1)
        no_action = lambda *args: None
        print("\tTesting the vocoder...")
        vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)

        print("All test passed! You can now synthesize speech.\n\n")
        self.encoder = encoder
        self.synthesizer = synthesizer
        self.vocoder = vocoder

    def synthesize_clone(self, reference, text):
        preprocessed_wav = self.encoder.preprocess_wav(reference)
        print("Loaded file succesfully")
        embed = self.encoder.embed_utterance(preprocessed_wav)
        print("Created the embedding")
        texts = [text]
        embeds = [embed]
        specs = self.synthesizer.synthesize_spectrograms(texts, embeds)
        spec = specs[0]
        print("Created the mel spectrogram")
        print("Synthesizing the waveform:")
        generated_wav = self.vocoder.infer_waveform(spec)

        return generated_wav

    def synthesizer_for(self, reference):
        preprocessed_wav = self.encoder.preprocess_wav(reference)
        print("Loaded file succesfully")
        embed = self.encoder.embed_utterance(preprocessed_wav)
        print("Created the embedding")

        def ref_synthesizer(text):
            texts = [text]
            embeds = [embed]
            specs = self.synthesizer.synthesize_spectrograms(texts, embeds)
            spec = specs[0]
            print("Created the mel spectrogram")
            print("Synthesizing the waveform:")
            generated_wav = self.vocoder.infer_waveform(spec)

            return generated_wav

        return ref_synthesizer


# clone_tts = CloneTTS()
# clone_tts.load_models()

# tts_model.synthesize_clone(ref_sample,text.strip())


def repl(tts_model, ref_sample_path):
    player = player_gen()

    def loop():
        text = input("tts >")
        data = tts_model.synthesize_clone(ref_sample, text.strip())
        player(data)

    return loop


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-r",
        "--ref_sample_path",
        type=Path,
        default="./OSR_us_000_0061_8k.wav",
        help="Path to a reference sample",
    )
    args = parser.parse_args()
    tts_model = TTSModel()
    # clone_tts = CloneTTS(**vars(args))
    clone_tts = CloneTTS()
    clone_tts.load_models()
    interactive_loop = repl(clone_tts, ref_sample)
    while True:
        interactive_loop()
