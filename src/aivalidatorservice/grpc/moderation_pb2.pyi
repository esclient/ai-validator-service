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

class ToxicityLevel(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TOXICITY_LEVEL_UNSPECIFIED: _ClassVar[ToxicityLevel]
    TOXICITY_LEVEL_NONE: _ClassVar[ToxicityLevel]
    TOXICITY_LEVEL_MODERATE: _ClassVar[ToxicityLevel]
    TOXICITY_LEVEL_SEVERE: _ClassVar[ToxicityLevel]
OBJECT_TYPE_UNSPECIFIED: ObjectType
OBJECT_TYPE_MOD_DESCRIPTION: ObjectType
OBJECT_TYPE_COMMENT_TEXT: ObjectType
OBJECT_TYPE_USER_NAME: ObjectType
TOXICITY_LEVEL_UNSPECIFIED: ToxicityLevel
TOXICITY_LEVEL_NONE: ToxicityLevel
TOXICITY_LEVEL_MODERATE: ToxicityLevel
TOXICITY_LEVEL_SEVERE: ToxicityLevel

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
    __slots__ = ("level",)
    LEVEL_FIELD_NUMBER: _ClassVar[int]
    level: ToxicityLevel
    def __init__(self, level: _Optional[_Union[ToxicityLevel, str]] = ...) -> None: ...
