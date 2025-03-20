#!/bin/bash

for pod in $(kubectl get pods --no-headers -o custom-columns=":metadata.name")
do
  podStatus=$(kubectl get pod $pod --no-headers -o custom-columns=":status.phase")
  if [[ $podStatus != *"Running"* && $podStatus != *"Pending"* && $podStatus != *"ContainerCreating"* ]]; then
    kubectl delete pod $pod
  fi
done
