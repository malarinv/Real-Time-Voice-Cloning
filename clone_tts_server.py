# -*- coding: utf-8 -*-
import grpc
import time
from concurrent import futures
from clone_tts import CloneTTS
from clone_tts_pb2 import CloneResponse
from clone_tts_pb2_grpc import CloneTTSServiceServicer
import clone_tts_pb2_grpc


class CloneTTSService(CloneTTSServiceServicer):
    def __init__(self):
        self.clone_tts_model = CloneTTS()
        self.clone_tts_model.load_models()

    def Synthesize(self, request, context):
        # while True:
        reference_speech = request.ref_speech
        input_text = request.text
        speech_response = self.clone_tts_model.synthesize_clone(reference_speech, input_text)
        return tts_pb2.CloneResponse(speech=speech_response)


def main():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    clone_tts_service = CloneTTSService()
    add_ServerServicer_to_server(clone_tts_service, server)
    server.add_insecure_port("localhost:50060")
    server.start()
    print("TTSServer started!")

    try:
        while True:
            time.sleep(10000)
    except KeyboardInterrupt:
        server.start()
        # server.stop(0)


if __name__ == "__main__":
    main()
