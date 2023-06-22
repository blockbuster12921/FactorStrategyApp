#!/bin/bash

if [ ! -f "$1" ]; then
	echo "Expected usage: bash restore.sh backup.gz"
	exit
fi

docker-compose exec -T mongo mongorestore --drop --gzip --archive < $1
