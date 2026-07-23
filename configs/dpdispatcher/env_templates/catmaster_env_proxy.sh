#!/usr/bin/env bash
# Copy to a site-owned remote path and source it before GPU provider environments.
# Remove this script from resource source_list when the compute host needs no proxy.
set -euo pipefail

export HTTP_PROXY="<REMOTE_HTTP_PROXY_URL>"
export HTTPS_PROXY="<REMOTE_HTTPS_PROXY_URL>"
export SOCKS_PROXY="<REMOTE_SOCKS_PROXY_URL>"
export NO_PROXY="127.0.0.1,localhost,::1"

# Some HTTP clients inspect only lowercase proxy variables.
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export no_proxy="${NO_PROXY}"
