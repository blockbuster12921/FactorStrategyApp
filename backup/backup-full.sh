#!/bin/bash

hour=$(TZ=Europe/London date '+%H')
if [ "$hour" != "08" ]
then
    printf "Full backup not required (hour = %s)\n" "$hour"
    exit 0
fi

cd /backups
mkdir -p full
cd ./full

ts=$(date '+%Y-%m-%dT%H%M%SZ')
backup_filename="$ts-mongodump.gz"

printf "%s: creating full backup %s in %s\n" "$ts" "$backup_filename" "$(pwd)"

mongodump --host "mongo:27017" --db FactorDBv2 --gzip --archive=$backup_filename

printf "%s: removing backups and log files older than 14 days\n" "$(date '+%Y-%m-%dT%H%M%SZ')"
find ./ -type f -mtime +14 -delete
