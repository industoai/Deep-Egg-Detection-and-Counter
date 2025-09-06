#!/bin/bash -l
set -e
if [ "$#" -eq 0 ]; then
  exec egg_detection_counter --help
else
  exec "$@"
fi
