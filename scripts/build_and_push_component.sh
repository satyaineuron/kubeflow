#!bin/bash

path=$1

repo=$2

aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 566373416292.dkr.ecr.ap-south-1.amazonaws.com

docker build -t 566373416292.dkr.ecr.ap-south-1.amazonaws.com/$repo:latest $path/

docker push 566373416292.dkr.ecr.ap-south-1.amazonaws.com/$repo:latest