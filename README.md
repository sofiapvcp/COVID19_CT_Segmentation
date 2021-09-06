# Segmentation of COVID-19 Lesions in CT Images

### Overview

The worldwide pandemic caused by the new coronavirus (COVID-19) has encouraged the development of multiple computer-aided diagnosis systems to automate daily clinical tasks, such as abnormality detection and classification. Among these tasks, the segmentation of COVID lesions is of high interest to the scientific community, enabling further lesion characterization. Automating the segmentation process can be a useful strategy to provide a fast and accurate second opinion to the physicians, and thus increase the reliability of the diagnosis and disease stratification. 

The current work explores a CNN-based approach to segment multiple COVID lesions. It includes the implementation of a **U-Net structure with a ResNet34 encoder** able to deal with the highly imbalanced nature of the problem, as well as the great variability of the COVID lesions, namely in terms of size, shape, and quantity. This **2D approach** yields a Dice score of 64.1%, when evaluated on the publicly available [**COVID-19-20 Lung CT Lesion Segmentation GrandChallenge data set**](https://covid-segmentation.grand-challenge.org/Data/).

### How to Use

#### Requirements
This code was developed using a Keras framework with a Tensorflow backend. The file with all the requirements is included in the repository.

#### File Structure


### Credits

If you use this code, please cite the following publication: 
**Rocha, Joana, et al. "Segmentation of COVID-19 Lesions in CT Images." 2021 IEEE EMBS International Conference on Biomedical and Health Informatics (BHI). IEEE, 2021.
https://ieeexplore.ieee.org/document/9508520**
