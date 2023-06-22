#!/bin/bash

cd /backups
mkdir -p mini
cd ./mini

ts=$(date '+%Y-%m-%dT%H%M%SZ')
backup_filename="$ts-mongodump.gz"

printf "%s: creating mini backup %s in %s\n" "$ts" "$backup_filename" "$(pwd)"

mongodump --host "mongo:27017" --db FactorDBv2 --excludeCollection=ProjectDriverParamSelectionState --excludeCollection=ProjectDataInfo --excludeCollection=DatasetReturnsData --excludeCollection=DatasetFactorData --excludeCollection=ProjectFactorGenerateState --excludeCollection=ProjectFactorCombinations --excludeCollection=ProjectFactorExpectedReturns --excludeCollection=ProjectFactorStrategyResults --excludeCollection=ProjectFactorCombinationStatus --excludeCollection=ProjectStockSelectionState --excludeCollection=ProjectRunState --gzip --archive=$backup_filename

printf "%s: removing backups older than 7 days\n" "$(date '+%Y-%m-%dT%H%M%SZ')"
find ./ -type f -mtime +7 -delete
