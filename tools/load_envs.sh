#!/bin/bash
set -e

if [ "$ENV" = "prod" ] && [ -f .env ]; then
    export $(grep -v '^#' .env | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | sed 's/"//g' | sed "s/'//g" | xargs)
fi

CONFIG="configs/${ENV}.yaml"
[ ! -f "$CONFIG" ] && echo "ERROR: Config not found: $CONFIG" >&2 && exit 1

eval "$(yq eval 'to_entries | .[] | "export " + .key + "='\''" + (.value | tostring) + "'\''"' "$CONFIG")"

if [ "$ENV" = "dev" ]; then
    exec "$@"
fi

VAULT_REFS_COUNT=$(env | grep -c '{vault:' || true)

if [ "$VAULT_REFS_COUNT" -eq 0 ]; then
    exec "$@"
fi

if [ -z "$VAULT_ADDR" ] || [ -z "$VAULT_TOKEN" ]; then
    echo "ERROR: Config contains vault references but VAULT_ADDR or VAULT_TOKEN is not set" >&2
    exit 1
fi

for var in $(env | grep '{vault:' | cut -d= -f1); do
    value="${!var}"
    
    while [[ "$value" =~ \{vault:([^}]+)\} ]]; do
        ref="${BASH_REMATCH[1]}"
        path="${ref%#*}"
        key="${ref##*#}"
        
        vault_mount="${VAULT_MOUNT_POINT:-secret}"
        vault_url="${VAULT_ADDR}/v1/${vault_mount}/data/${path}"
        
        set +e
        vault_response=$(curl -s -S -w "\n%{http_code}" \
            --header "X-Vault-Token: $VAULT_TOKEN" \
            "$vault_url" 2>&1)
        curl_exit_code=$?
        set -e
        
        if [ $curl_exit_code -ne 0 ]; then
            echo "ERROR: Failed to fetch secret from Vault: $vault_url" >&2
            echo "Curl response: $vault_response" >&2
            exit 1
        fi
        
        http_code=$(echo "$vault_response" | tail -n1)
        vault_response=$(echo "$vault_response" | sed '$d')
        
        if [ "$http_code" != "200" ]; then
            echo "ERROR: Vault API returned HTTP $http_code for $vault_url" >&2
            exit 1
        fi
        
        secret=$(echo "$vault_response" | jq -r ".data.data.$key // .data.$key")
        
        if [ "$secret" = "null" ] || [ -z "$secret" ]; then
            echo "ERROR: Secret key '$key' not found at path '$path'" >&2
            exit 1
        fi
        
        value="${value/\{vault:$ref\}/$secret}"
    done
    
    export "$var=$value"
done

exec "$@"
