# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: clone_tts.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='clone_tts.proto',
  package='clone_tts',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x0f\x63lone_tts.proto\x12\tclone_tts\".\n\nCloneInput\x12\x0c\n\x04text\x18\x01 \x01(\t\x12\x12\n\nref_speech\x18\x02 \x01(\x0c\"\x1f\n\rCloneResponse\x12\x0e\n\x06speech\x18\x01 \x01(\x0c\x32R\n\x0f\x43loneTTSService\x12?\n\nSynthesize\x12\x15.clone_tts.CloneInput\x1a\x18.clone_tts.CloneResponse\"\x00\x62\x06proto3')
)




_CLONEINPUT = _descriptor.Descriptor(
  name='CloneInput',
  full_name='clone_tts.CloneInput',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='text', full_name='clone_tts.CloneInput.text', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ref_speech', full_name='clone_tts.CloneInput.ref_speech', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=30,
  serialized_end=76,
)


_CLONERESPONSE = _descriptor.Descriptor(
  name='CloneResponse',
  full_name='clone_tts.CloneResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='speech', full_name='clone_tts.CloneResponse.speech', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=78,
  serialized_end=109,
)

DESCRIPTOR.message_types_by_name['CloneInput'] = _CLONEINPUT
DESCRIPTOR.message_types_by_name['CloneResponse'] = _CLONERESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

CloneInput = _reflection.GeneratedProtocolMessageType('CloneInput', (_message.Message,), {
  'DESCRIPTOR' : _CLONEINPUT,
  '__module__' : 'clone_tts_pb2'
  # @@protoc_insertion_point(class_scope:clone_tts.CloneInput)
  })
_sym_db.RegisterMessage(CloneInput)

CloneResponse = _reflection.GeneratedProtocolMessageType('CloneResponse', (_message.Message,), {
  'DESCRIPTOR' : _CLONERESPONSE,
  '__module__' : 'clone_tts_pb2'
  # @@protoc_insertion_point(class_scope:clone_tts.CloneResponse)
  })
_sym_db.RegisterMessage(CloneResponse)



_CLONETTSSERVICE = _descriptor.ServiceDescriptor(
  name='CloneTTSService',
  full_name='clone_tts.CloneTTSService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=111,
  serialized_end=193,
  methods=[
  _descriptor.MethodDescriptor(
    name='Synthesize',
    full_name='clone_tts.CloneTTSService.Synthesize',
    index=0,
    containing_service=None,
    input_type=_CLONEINPUT,
    output_type=_CLONERESPONSE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_CLONETTSSERVICE)

DESCRIPTOR.services_by_name['CloneTTSService'] = _CLONETTSSERVICE

# @@protoc_insertion_point(module_scope)
