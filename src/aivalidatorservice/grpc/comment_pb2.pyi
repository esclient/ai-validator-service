import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CommentStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    COMMENT_STATUS_UNSPECIFIED: _ClassVar[CommentStatus]
    COMMENT_STATUS_DELETED: _ClassVar[CommentStatus]
    COMMENT_STATUS_HIDDEN: _ClassVar[CommentStatus]
    COMMENT_STATUS_ON_MODERATION: _ClassVar[CommentStatus]
COMMENT_STATUS_UNSPECIFIED: CommentStatus
COMMENT_STATUS_DELETED: CommentStatus
COMMENT_STATUS_HIDDEN: CommentStatus
COMMENT_STATUS_ON_MODERATION: CommentStatus

class Comment(_message.Message):
    __slots__ = ("id", "author_id", "text", "created_at", "edited_at")
    ID_FIELD_NUMBER: _ClassVar[int]
    AUTHOR_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    EDITED_AT_FIELD_NUMBER: _ClassVar[int]
    id: int
    author_id: int
    text: str
    created_at: _timestamp_pb2.Timestamp
    edited_at: _timestamp_pb2.Timestamp
    def __init__(self, id: _Optional[int] = ..., author_id: _Optional[int] = ..., text: _Optional[str] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., edited_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class CreateCommentRequest(_message.Message):
    __slots__ = ("mod_id", "author_id", "text")
    MOD_ID_FIELD_NUMBER: _ClassVar[int]
    AUTHOR_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    mod_id: int
    author_id: int
    text: str
    def __init__(self, mod_id: _Optional[int] = ..., author_id: _Optional[int] = ..., text: _Optional[str] = ...) -> None: ...

class CreateCommentResponse(_message.Message):
    __slots__ = ("comment_id",)
    COMMENT_ID_FIELD_NUMBER: _ClassVar[int]
    comment_id: int
    def __init__(self, comment_id: _Optional[int] = ...) -> None: ...

class GetCommentsRequest(_message.Message):
    __slots__ = ("mod_id",)
    MOD_ID_FIELD_NUMBER: _ClassVar[int]
    mod_id: int
    def __init__(self, mod_id: _Optional[int] = ...) -> None: ...

class GetCommentsResponse(_message.Message):
    __slots__ = ("mod_id", "comments")
    MOD_ID_FIELD_NUMBER: _ClassVar[int]
    COMMENTS_FIELD_NUMBER: _ClassVar[int]
    mod_id: int
    comments: _containers.RepeatedCompositeFieldContainer[Comment]
    def __init__(self, mod_id: _Optional[int] = ..., comments: _Optional[_Iterable[_Union[Comment, _Mapping]]] = ...) -> None: ...

class SetStatusRequest(_message.Message):
    __slots__ = ("comment_id", "status")
    COMMENT_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    comment_id: int
    status: CommentStatus
    def __init__(self, comment_id: _Optional[int] = ..., status: _Optional[_Union[CommentStatus, str]] = ...) -> None: ...

class SetStatusResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...

class EditCommentRequest(_message.Message):
    __slots__ = ("comment_id", "text")
    COMMENT_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    comment_id: int
    text: str
    def __init__(self, comment_id: _Optional[int] = ..., text: _Optional[str] = ...) -> None: ...

class EditCommentResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...
