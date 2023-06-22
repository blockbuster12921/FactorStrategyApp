#!/bin/bash

if [ -z $1 ]; then
	printf "Usage: make_release.sh <version>\n"
	exit
fi

version=$1

base_dir="FactorStrategyApp_$1"
echo "Making release in \"$base_dir\"..."
rm -rf ./$base_dir
mkdir ./$base_dir
cd $base_dir

dir="FactorStrategyApp"
if [ -d $dir ] 
then
    echo "Error: Directory $dir already exists"
	exit 1 
fi
git clone --single-branch --branch master git@ssh.dev.azure.com:v3/ojoloan/SKFactorStrategy/SKFactorStrategy $dir
rm -rf ./$dir/data
rm -rf ./$dir/.git
rm ./$dir/.gitignore

cd ..
zip -r ${base_dir}.zip $base_dir
rm -rf $base_dir
