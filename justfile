set windows-shell := ["sh", "-c"]
set dotenv-load := true

COMMON_JUST_URL := 'https://raw.githubusercontent.com/esclient/tools/refs/heads/main/python/common.just'
LOAD_ENVS_URL := 'https://raw.githubusercontent.com/esclient/tools/refs/heads/main/load_envs.sh'

COMMENT_PROTO_TAG := 'v0.0.17'
COMMENT_PROTO_NAME := 'comment.proto'
COMMENT_TMP_DIR := '.proto'
SOURCE := 'commentservice'
OUT_DIR := 'src/' + SOURCE + '/grpc'

MODERATION_PROTO_TAG := 'v0.1.3'
MODERATION_PROTO_NAME := 'moderation.proto'
MODERATION_TMP_DIR := '.proto'
SERVICE_NAME := 'moderation'

MKDIR_TOOLS := 'mkdir -p tools'

FETCH_COMMON_JUST := 'curl -fsSL ' + COMMON_JUST_URL + ' -o tools/common.just'
FETCH_LOAD_ENVS := 'curl -fsSL ' + LOAD_ENVS_URL + ' -o tools/load_envs.sh'

import? 'tools/common.just'

default:
    @just --list

fetch-tools:
    {{ MKDIR_TOOLS }}
    {{ FETCH_COMMON_JUST }}
    {{ FETCH_LOAD_ENVS }}
