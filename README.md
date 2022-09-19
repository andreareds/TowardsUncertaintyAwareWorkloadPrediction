# Uncertainty-Aware Workload Prediction in Cloud Computing

Predicting future demand in Cloud Computing is essential for managing a Cloud data centre and guaranteeing a minimum Quality of Service (QoS) level to the customers. Modelling the uncertainty of the future resource demand leads to a more accurate quality of the prediction and a reduction of the waste due to overallocation. Generally, many cluster cells share common patterns, such as workload peaks over daylight hours, that can be exploited to improve prediction accuracy.

In this paper, we propose Bayesian deep learning models trained with multiple datasets to predict the distribution of the future demand of processing units and memory and model the uncertainty in the context of cloud workload prediction. We compare them to the same models trained with one single dataset for both univariate and bivariate cases, showing that more data help the model to generalize and make it more accurate. Moreover, we provide an ablation study of a bivariate model trained with multiple datasets and compare the results to the univariate counterpart to investigate which components are more affecting the accuracy of the prediction and their impact on the QoS. Finally, we investigate the effect of the transfer learning approach in the context of cloud workload prediction.
Results show the benefits of training the networks with multiple datasets, without having seen the same data before for both the univariate and bivariate cases, with the advantages of using pretrained models to predict related but unseen time series.
Contextually, we preprocessed in a consistent and detailed way 12 datasets from the public Google Cloud and Alibaba Cloud Computing systems' traces that we used for the experiments and made them available to facilitate the research in this field. 

## Python Dependencies
* arch                      5.1.0
* keras                     2.8.0
* matplotlib                3.3.4
* numpy                     1.21.5
* pandas                    1.2.3
* python                    3.7.9
* statsmodels               0.12.2
* talos                     1.0.2 
* tensorflow                2.8.0
* tensorflow-gpu            2.8.0
* tensorflow-probability    0.14.0

## Project Structure
* **hyperparams**: contains for each deep learning model the list of optimal hyperparameters found with Talos.
* **img**: contains output plot for predictions, models and loss function.
* **models**: contains the definition of statistical and deep learning models. One can train the model from scratch using the optimal parameters found with Talos, look for the optimal hyperparameters by changing the search space dictionary or load a saved model and make new forecasts.
* **param**: contains for each statistical model the list of optimal parameters found.
* **res**: contains the results of the prediction
* **saved_data**: contains the preprocessed datasets.
* **saved_models**: contains the model saved during the training phase.
* **time**: contains measurements of the time for training, fine-tuning and inference phases.
* **util**: contains useful methods for initialising the datasets, plotting and saving the results.

#### Train LSTM

```bash
python lstm_training.py
```

#### Train HBNN

```bash
python hbnn_training.py
```

#### Train LSTMD

```bash
python lstmd_training.py
```
