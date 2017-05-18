#!/bin/bash
for filename in "$1"/*.txt; do
	python Fluidity.py $filename "$2" "$3"
done
