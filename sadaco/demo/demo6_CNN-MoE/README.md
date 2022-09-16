# CNN-MoE Based Framework for Classification of Respiratory Anomalies and Lung Disease Detection

This is an unofficial PyTorch implementation of "[CNN-MoE framework for classification of respiratory anomalies and lung disease detection](https://ieeexplore.ieee.org/abstract/document/9372748)" by L. Pham _et al._, IEEE Journal of Biomedical and Health Informatics, August 2021.

## About The Paper
The authors present a CNN-MoE framework for auscultation analysis. The framework begins with front-end feature extraction that transforms input sound into a spectrogram representation. Then, they use a VGG-like backend network to extract features and classify the spectrogram into categories of repiratory anomaly cycles or diseases. In their paper, they conducted several experiments on the ICBHI'17 dataset and found that this CNN-MoE framework outperforms current state-of-the-art methods. 

## About This Demo
In this demo, we use the CNN-MoE framework to classify respiratory anomalies in Fraiwan dataset. 

### Training
To train the model on the Fraiwan dataset, please run the following command: 
```python
python3 demo_CNNMoE.py --input_data <path_to_data> \
                       --master_config <path_to_master_config> \
                       --model_config <path_to_model_config> \
```
Or use `--help` to see all the options.

#### Hyperparameters
Please refer to the `configs_/` folder the configurations for training, feature extraction, and model. 

### Evaluation
We evaluate the model on the Fraiwan dataset on four metrics, which are:
1. Accuracy: The accuracy of the model on the test set.
2. Sensitivity: The ability of the model to detect abnormal cycles.
3. Specificity: The ability of the model to detect normal cycles.
4. Score: The average of sensitivity and specificity.

With the default implementation settings, the model achieved an accuracy of **something**, average score of **something**, sensitivity of **something**, and specificity of **something**.