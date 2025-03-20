#!/bin/bash

directory="radar-sim"

for file in "$directory"/*.yaml; do
    filename=$(basename "$file")
    echo "Running kubectl create -f $directory/$filename"
    kubectl create -f "$directory/$filename"
done