#!/bin/bash

if ! command -v uv >/dev/null 2>&1; then
uv venv --quiet --no-cache --clear --python 3.13
  wget -O "$installer" https://astral.sh/uv/install.sh
  sh "$installer"
fi
uv venv --quiet --no-cache --clear --python 3.13
source .venv/bin/activate