#!/bin/sh
for num in $(seq 1 $PEXECS)
do
	$1/release/static-web-server -p 8787 -d $1/www/ &
	http_pid=$!

	sleep 1
	$2 -c $3 -t $4 -d $5 --latency -s benchmark_collector.lua $6
	kill $http_pid
done
