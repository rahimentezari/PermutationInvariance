# PermutationInvarience
This repository includes the codes for " The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks" submitted to ICLR 2022.

### Working with Caliban
Most experiments in this repositoty were done using [Caliban](https://github.com/google/caliban). Caliban is a tool for developing research workflow and notebooks in an isolated Docker environment and submitting those isolated environments to Google Compute Cloud.
Basically you can use the commands in run.sh for different experiments. Each run will load the hyperparameters from [config.json](https://github.com/rahimentezari/DataDisributionTransferLearning/blob/main/config.json) and save results in the Google Bucket.
Below is a short step-by-step how to run Caliban on GCP:
1. sudo apt-get install python3 python3-venv python3-pip
2. sudo usermod -a -G docker ${USER}
3. Install Docker:
Note: check if docker is already installed:
sudo apt-get install -y nvidia-docker2
If not continue:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker
4. sudo pkill -SIGHUP dockerd
5. python3 -m pip install --user pipx
6. python3 -m pipx ensurepath
7. source ~/.bashrc (or re-login for the PATH changes to take effect)
8. pipx install caliban
> To check if all is well, run
caliban --help

### Setting up Google Cloud for Caliban
9. Give the account owner the name of the account:
Go to vm details> API and identity management
> Service account 
Add the Service account($$$@developer.gserviceaccount.com) as an owner to the IAMadmin in google console.

10. Also add this to the bucket as storage object admin if you are using Google Bucket

11. gcloud init
- Select the account
- Set default zone to some zone e.g. europe-west4-a (number 14)
12. Add the following lines to the end of “~/.bashrc”
export REGION="your zone e.g. europe-west4 "
export PROJECT_ID="your project ID"

source ~/.bashrc

> Test your Environment: gcloud auth list
13. [Follow these steps to get a JSON file for credentials](https://cloud.google.com/iam/docs/creating-managing-service-account-keys#iam-service-account-keys-create-console)
14. Move the json file to a path
15. Add the following to the end of “~/.bashrc”:
export GOOGLE_APPLICATION_CREDENTIALS=path to the JSON file
16. source ~/.bashrc

Then you can either run caliban [locally](https://caliban.readthedocs.io/en/stable/cli/caliban_run.html) or on the [cloud using GCP Training jobs](https://caliban.readthedocs.io/en/stable/cli/caliban_cloud.html)
