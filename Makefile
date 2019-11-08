grpc_stub:
	python -m grpc_tools.protoc -I. --python_out=./ --grpc_python_out=./ --proto_path . clone_tts.proto

start_server:
	python clone_tts_server.py
