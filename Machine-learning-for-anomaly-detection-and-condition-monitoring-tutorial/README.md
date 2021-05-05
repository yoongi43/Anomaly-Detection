# Machine learning for anomaly detection and condition monitoring tutorial
This is an implementation of anomaly detection tutorial of NASA bearing dataset.

## Citation
Code : https://towardsdatascience.com/machine-learning-for-anomaly-detection-and-condition-monitoring-d4614e7de770

Dataset : J. Lee, H. Qiu, G. Yu, J. Lin, and Rexnord Technical Services (2007). IMS, University of Cincinnati. "Bearing Data Set", NASA Ames Prognostics Data Repository (http://ti.arc.nasa.gov/project/prognostic-data-repository), NASA Ames Research Center, Moffett Field, CA

## Requirement
python==3.8


## Data
From Prognostics Data Repository

Downloaded from https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

From above page, download 4th dataset : Bearing Data Set

Files are unzipped and saved at './data'

"4th_test" dataset of Bearing Data Set has another 'txt' folder in the path. All the files should be out of txt folder and put in '4th_text' folder


## Description
These datasets are experiments on bearings provided by Center fo Intelligent Maintenance Systems(IMS), University of Cincinnati.

Each dataset consists of individual files that are 1-second vibration signal snapshots recorded at
specific intervals (with the sampling rate of 20kHz).
Thus, each file consists of $ 20480 \approx 20k $ points

The file name indicates when the data was collected.

Each record (row) in the data file is a data point, which describes a test-to-failure experiment.

Therefore, most of the former part of each dataset represent normal operating conditions,
considered to be training set and latter would be test set.

I divided their ratio by 8 to 2.

### load.py
It reads training set and test set from directory. 

There are also an additional option(proportion of training set, showing the plot of dataset, normalization etc).

For each file, all the data are taken an absolute value and averaged.

### model_PCA
PCA class for dataset.

PCA is one of the dimensionality reduction methods.

For example, 2nd dataset has (984, 4) data, and we split it into training set (787, 4) and test set (197, 4).
PCA model reduces dimension of training set into (787, 2) (4 to 2).

Here, we use Mahalanobis distance to measure the distance between a point and a distribution.

That is, we calculate covariance matrix and Mahalanobis distance of training set.
While for the test set, Mahalanobis distance is calculated based on covariance matrix of training set.

Then, data of test dataset would be outliers if their distance is larger than specific threshold driven from Mahalanobis distance of training set.

### model_Autoencoder.py
Here, we use an autoencoder to 'compress' the sensor readings to a low dimensional representation, which captures the correlations and interactions between the various variables.
(Essentially the same principle as the PCA model, but her we also allow for non-linearity among the input variables).
 
The main idea is that as the monitored equipment degrades, this should affect the interaction between the variables.
As this happens, one will start to see an increased error in the networks re-construction of th input variables.


Model consists of 3 fully connected network, which has 10, 2, 10 nodes each.

Encoder learns interactions between the various variables and compress it to second layer with 2 nodes.

Decoder learns to decompress the compressed feature to original size.

For model trained with MSE(Mean Square Error) loss function, input the training set and calculate MAE(Mean Absolute Eror).
From the distribution of MAE of training set, define threshold of anomaly.
Then, calculate the MAE of output of test set and flag it as anomaly if it has MAE 
above threshold.

Model is trained with Adam optimizer.

