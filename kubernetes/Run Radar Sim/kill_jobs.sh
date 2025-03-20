#!/bin/bash

directory="radar-sim"

for file in "$directory"/*.yaml; do 
    filename=$(basename "$file")
    echo "Running kubectl delete -f $directory/$filename" 
    kubectl delete -f "$directory/$filename"
done