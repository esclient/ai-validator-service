from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ObjectType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OBJECT_TYPE_UNSPECIFIED: _ClassVar[ObjectType]
    OBJECT_TYPE_MOD_DESCRIPTION: _ClassVar[ObjectType]
    OBJECT_TYPE_COMMENT_TEXT: _ClassVar[ObjectType]
    OBJECT_TYPE_USER_NAME: _ClassVar[ObjectType]
OBJECT_TYPE_UNSPECIFIED: ObjectType
OBJECT_TYPE_MOD_DESCRIPTION: ObjectType
OBJECT_TYPE_COMMENT_TEXT: ObjectType
OBJECT_TYPE_USER_NAME: ObjectType

class ModerateObjectRequest(_message.Message):
    __slots__ = ("id", "type", "text")
    ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    id: int
    type: ObjectType
    text: str
    def __init__(self, id: _Optional[int] = ..., type: _Optional[_Union[ObjectType, str]] = ..., text: _Optional[str] = ...) -> None: ...

class ModerateObjectResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...
