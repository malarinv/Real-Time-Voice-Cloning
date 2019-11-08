from clone_tts import CloneTTS
import argparse
from pathlib import Path
import wave
import numpy as np

OUTPUT_SAMPLE_RATE = 16000


def float2pcm(sig, dtype="int16"):
    """Convert floating point signal with a range from -1 to 1 to PCM.
    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.
    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
    Parameters
    ----------
    sig : array_like
        Input array, must have floating point type.
    dtype : data type, optional
        Desired (integer) data type.
    Returns
    -------
    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        *dtype*.
    See Also
    --------
    pcm2float, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != "f":
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in "iu":
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


def synthesize_corpus(corpus_path=Path("corpus.txt"), output_dir=Path("./out_dir")):
    clone_tts = CloneTTS()
    clone_tts.load_models()
    tts_model = clone_tts.synthesizer_for("./nicole_sample.wav")
    output_dir.mkdir(exist_ok=True)
    for (i, line) in enumerate(open(str(corpus_path)).readlines()):
        print(f'synthesizing... "{line.strip()}"')
        data_arr = tts_model(line.strip())
        data = float2pcm(data_arr.astype(float)).tobytes()
        out_file = str(output_dir / Path(str(i) + ".wav"))
        with wave.open(out_file, "w") as out_file_h:
            out_file_h.setnchannels(1)  # mono
            out_file_h.setsampwidth(2)  # pcm int16 2bytes
            out_file_h.setframerate(OUTPUT_SAMPLE_RATE)
            out_file_h.writeframes(data)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-c",
        "--corpus_path",
        type=Path,
        default="./corpus.txt",
        help="Path to a corpus file",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default="./synth",
        help="Path to a output directory",
    )
    args = parser.parse_args()
    synthesize_corpus(**vars(args))


if __name__ == "__main__":
    main()
