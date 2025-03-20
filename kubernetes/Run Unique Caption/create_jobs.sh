#!/bin/bash

directory="unique-caption"

for file in "$directory"/*.yaml; do
    filename=$(basename "$file")
    echo "Running kubectl create -f $directory/$filename"
    kubectl create -f "$directory/$filename"
done