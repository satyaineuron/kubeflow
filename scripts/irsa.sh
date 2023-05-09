#!bin/bash

# arn:aws:iam::566373416292:policy/ecom_kubeflow_policy - Create this policy with s3 and ecr full access

eksctl create iamserviceaccount --name kube-imdb-sa --cluster kubeflow --namespace kubeflow-user-example-com --role-name kube_ecom_role --attach-policy-arn=arn:aws:iam::566373416292:policy/kubeflow-imdb-admin --approve --override-existing-serviceaccounts

kubectl create clusterrolebinding kube-imdb-sa-cluster-role-binding --clusterrole=admin --serviceaccount=kubeflow-user-example-com:kube-imdb-sa
 