#!/bin/zsh
# Stops the given process when its physical footprint passes the limit in GB.
pid=$1; limit=$2
while kill -0 $pid 2>/dev/null; do
  f=$(vmmap --summary $pid 2>/dev/null | awk '/^Physical footprint:/{print $3}')
  case $f in *G) v=${f%G}; if (( v > limit )); then echo "GUARD stop at $f"; kill $pid; exit 0; fi;; esac
  sleep 20
done
