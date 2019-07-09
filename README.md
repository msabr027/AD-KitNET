# AD-KitNET
Anomaly detection tool based on ensemble of autoencoders, From:

Yisroel Mirsky, Tomer Doitshman, Yuval Elovici, and Asaf Shabtai, "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection", Network and Distributed System Security Symposium 2018 (NDSS'18)
# What is it ?

Adapted from the research below, it's a lighweight, unsupervised and agnostic system for detecting any kind of anomaly on any type of data.
The tool is based on three algorithms groups:Feature mapping, Ensembles of Autoencoder, Voting Layer(output layer).
In addition the tool can be used on data stream, images, marketing databases, fraud data.

The main advantage is the fact that no label is needed on data. You present the raw data to the system, and it take care of everything.

To find out more about the original developpement on the tool follow this link : https://github.com/ymirsky/KitNET-py

# Setup

To be able to use the tool, the requirements.txt contains the necessary dependencies to use the tool.

```
pip install requirements.txt
```

### Enable GPU

To be able to use GPU acceleration with the tool, you have to install cuda. GPU is enabled by cuPY dependency. After installing CUDA, you have to detect which version of CUDA is installed :

```
nvcc --version
```

After you can install cuPY depending on the vesion of CUDA: 
```
(Binary Package for CUDA 8.0)
$ pip install cupy-cuda80

(Binary Package for CUDA 9.0)
$ pip install cupy-cuda90

(Binary Package for CUDA 9.1)
$ pip install cupy-cuda91

(Binary Package for CUDA 9.2)
$ pip install cupy-cuda92

(Binary Package for CUDA 10.0)
$ pip install cupy-cuda100

(Binary Package for CUDA 10.1)
$ pip install cupy-cuda101
```

# Tool usage

To use the tool, from the command line you can invoke the tool for training or scoring.

## Training

To be able to train the model the following command can be used, this will run the example related to fraud detection:

```
python train.py --file=./data/bank.csv --type=num --format=csv --sens=low --epoch=5 --gpu=False
```

Before that you have to put the data training data in the following directory : 

./data for numeric data

./data/iamges for images data

then use the following options to be able to run properly the model:

--file : identify where the data for training is stored (not forget to add at the end the file name and file extension.

--type : identify which type of data its, 2 choices from num or image (num stands for numeric)

--format : identify the file format  ONLY for numerical data (csv,hdf,excel,parquet,json)

--sens : identify the cutoff of the model. (it helps fix a threshold of detection). low means that you allow more false positive. high means you allow more false negative. med means between low and high.

--epoch : identify the number of epochs that will be performed by the model. Or the number of times the model will see the data. The default is set at 1.

--imgsize1 : identify the image size weight for the case. This option is valid for the case where type is image.

--imgsize2 : identify the image size height for the case. This option is valid for the case where type is image.

--gray : identify is the image is gray scale. This option is valid for the case where type is image. Default is set to false.

--gpu : identify if gpu acceleration will be used to train the model. Default is set to false.

## Scoring

To be able to evaluate the model, the following commanda can be used, this will run the example related to fraud detection:
Note that before evaluating (scoring), you need to train the model first.

```
python evaluate.py --file=./data/bank.csv --type=num --format=csv
```
For scoring the data you need to set up the following options :

--file : where the data to score is stored

--type : num or image

--format : for numerical to know if the data is csv, hdf, parquet, excel, json

## Examples

In the repo, you have three examples, to run:

./data/bank.csv for fraud detection in banking

./data/images for image anomaly detection on MNIST data

./data/iot.csv for internet of things and sensor data anomaly detection
