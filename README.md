# DeepPIM: Deep Neural Point-of-Interest Imputation Model
A point-of-interest (POI) is a specific location in which someone is interested. 
Users on Instagram, a mobile-based social network, share their experiences with text and photos, and link POIs to their posts.
POIs can be utilized to understand user preferences and behavior. 
However, POI information is not annotated in more than half of the total generated data. 
Therefore, it is necessary to automatically associate a post on Instagram with a POI.
In a previous study, a POI prediction model was trained on the POI-annotated Instagram data which includes user, text, and photo information.
However, this model has some limitations such as difficulty in handling a large amount of data and the high cost of feature engineering.
In addition, this model does not utilize posting time information which provides each POI's temporal characteristics.
In this paper, we propose a novel deep learning based time-aware POI prediction model that processes a large amount of data without feature engineering.
Our proposed model utilizes text, photo, user, and time information to predict correct POIs.
The experimental results show that our model significantly outperforms the existing state-of-the-art model.

## Model description
<p align="center">
<img src="/figures/model_description.png" width="400px" height="auto">
</p>
DeepPIM consists of two DNN layers, textual RNN and visual CNN layers, and two latent feature matrices for user and time embedding.

## Data set
Data set is available at [here](https://s3.amazonaws.com/poiprediction/instagram.tar.gz). The data set includes "train.txt", "validation.txt", "test.txt", and "visual_feature.npz". The "train.txt"  "validation.txt" "test.txt" files include the training, validation, and tesing data respectively. The data is represented in the following format:
```bash
<post_id>\t<user_id>\t<word_1 word_2 ... >\t<poi_id>\t<month>\t<weekday>\t<hour>
```

All post_id, user_id, word_id, and poi_id are anonymized. Photo information also cannot be distributed due to personal privacy problems. So we relase the converted visual features from the output of the FC-7 layer of [VGGNet](https://arxiv.org/pdf/1409.1556.pdf) used as the visual feature extractor. If you want to use other visual feature extractor, such as [GoogleNet](http://arxiv.org/abs/1602.07261), [ResNet](https://arxiv.org/abs/1512.03385), you could implement it on your source code. We use a pre-trained VGGNet16 by [https://github.com/machrisaa/tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg) The "visual_feature.npz" file contains the visual features where the i-th row denotes i-th post's features.

### statistics
<table style="align=center;">
<tr><td>number of total post</td><td>number of POIs</td><td>number of users</td><td>size of vocabulary</td></tr>
<tr><td>736,445</td><td>9,745</td><td>14,830</td><td>470,374</td></tr>
<tr><td>size of training set</td><td>size of validation set</td><td>size of test set</td></tr>
<tr><td>526,783</td><td>67,834</td><td>141,828</td></tr>
</table>

## Getting Started
The code that implements our proposed model is implemented for the above dataset, which includes pre-processd visual feature. If you want to use a raw image that is not pre-processed, implement VGGNet on your source code as visual CNN layer.

### Prerequisites
- python 2.7
- tensorflow r1.2.1

### Usage
```bash
git clone https://github.com/qnfnwkd/DeepPIM
cd DeepPIM
python train.py
```
