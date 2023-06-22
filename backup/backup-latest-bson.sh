#!/bin/bash

cd /backups
mkdir -p latest-bson
cd ./latest-bson

ts=$(date '+%Y-%m-%dT%H%M%SZ')
printf "%s: updating latest bson in %s\n" "$ts" "$(pwd)"

mongodump --host "mongo:27017" --db FactorDBv2 --excludeCollection=ProjectDriverParamSelectionState --excludeCollection=DatasetDataInfo --excludeCollection=DatasetReturnsData --excludeCollection=DatasetFactorData --excludeCollection=ProjectFactorGenerateState --excludeCollection=ProjectFactorCombinations --excludeCollection=ProjectFactorExpectedReturns --excludeCollection=ProjectFactorStrategyResults --excludeCollection=ProjectFactorCombinationStatus --excludeCollection=ProjectStockSelectionState --excludeCollection=ProjectRunState --gzip --out .
