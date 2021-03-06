# Udacity Deep Reinforcement Learning Nanodegree Project: Tennis

## Project Details

The objective of this project is to solve a modified version of the Unity ML-Agents task, `Tennis`, in which two agents
control rackets and are expected to collaborate in order to keep a ball in the play.  

### Specifics of the Task
* The observable state space consists of 8 feature variables, which correspond to the positions and velocities of the
ball and racket.  
* The action space consists of 2 continuous variables, which correspond to the movements toward or away from the net,
as well as a movement upward ("jump" motion).  
* The state consists of a 24 continuous features referring the location and trajectories of relevant objects.  Each 
agent receives its own state observation. 
* A reward of 0.1 is provided for successfully striking the ball over the net.  A reward of -0.01 is provided for
allowing the ball to strike the table or exit out of bounds.  
* The task is considered solved when the agents are capable of averaging 0.5 over the course of 100 episodes.  An
episode score is defined as the maximum total reward recorded from the two agents.  


## Installation

This repository runs within the provided docker environment. The base image upon which this
repository's docker image is built is freely available from my DockerHub,
`ccthompson82/drlnd:0.0.8`.  No downloads are necessary if the instructions below are followed.

### Dependencies
* Python 2.7 or Python 3.5
* Docker version 17 or later
    - [docker](https://docs.docker.com/install/)

## Setup the docker image

1. Update the data directory in the Makefile of this project's repository.  
    * Modify the environment variable definition on line 37.  It can be advantageous to mount a storage directory,
     though a default option would simply be `DATA_SOURCE=$(PWD)/data` to keep and track data locally in this
     repository.    

2. Setup the development environment in a Docker container with the following command:
    - `make init`

    This command gets the resources for training and testing, and then prepares the Docker image for the experiments.

## Launching the docker container

1. After creating the Docker image, run the following command.

- `make create-container`

    The above command creates a Docker container from the Docker image which we create with `make init`, and then
login to the Docker container.  This command needs to be run only once after creating the docker image.  After the
container hs been created with the command above, use the following command to enter the existing container: `make start-container`.

# Directory Structure

* Makefile - contains many targets such as create docker container or get input files.
* config - contains configuration files used in scripts
* data - store the model data files created in the experiments.  *NOTE*: This dir will be populated by running
    "make mount-prodmod" from inside the docker container with actual experiment model weights.  
* docker - contains Dockerfile.
* **model - [PRIMARY CODE DIR] contains all the model implementations, as well as hyperparams and params used at training
time.**  
* src - contains base objects and clients used for loading and utilising models
* notebook - jupyter notebooks
* scripts - contains generic scripts that train or evaluate a given model named in the `config/model.json` file.


# Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [cookiecutter-docker-science](https://docker-science.github.io/) project template.
