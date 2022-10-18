# Uncertainty-Aware Workload Prediction in Cloud Computing

Predicting future resource demand in Cloud Computing is essential for managing Cloud data centres and guaranteeing customers a minimum Quality of Service (QoS) level. Modelling the uncertainty of future demand improves the quality of the prediction and reduces the waste due to overallocation. In this paper, we propose univariate and bivariate Bayesian deep learning models to predict the distribution of future resource demand and its uncertainty. We design different training scenarios to train these models, where each procedure is a different combination of pretraining and fine-tuning steps on multiple datasets configurations. We also compare the bivariate model to its univariate counterpart training with one or more datasets to investigate how different components affect the accuracy of the prediction and impact the QoS. Finally, we investigate whether our models have transfer learning capabilities.
Extensive experiments show that pretraining with multiple datasets boosts performances while fine-tuning does not. Our models generalise well on related but unseen time series, proving transfer learning capabilities. Runtime performance analysis shows that the models are deployable in real-world applications. For this study, we preprocessed twelve datasets from real-world traces in a consistent and detailed way and made them available to facilitate the research in this field.

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

##### Model types:
* univariate
* bivariate
* multiunivariate
* multibivariate

#### Methods:

```bash
training_talos
```

Run the hyperparameters search via Talos optimization library

```bash
training
```

Train a model from scratch, given a set of hyperparameters

```bash
load_and_predict
```

Load a pretrained model and make the prediction

```bash
load_and_tune
```

Load a pretrained model and tune the network on the specified dataset
