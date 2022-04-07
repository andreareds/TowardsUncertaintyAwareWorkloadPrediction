# Uncertainty-Aware Workload Prediction in Cloud Computing

The prediction of future demand in Cloud Computing is essential for the management of a Cloud data centre and to guarantee a minimum Quality of Service (QoS) level to the customers. Modelling the uncertainty of the future resource demand leads to a more accurate quality of the prediction and a reduction of the waste due to overallocation. Generally, many cluster cells share common patterns such as workload peaks over daylight hours that can be exploited to improve the accuracy of the prediction.

In this paper, we propose Bayesian deep learning models that are trained with multiple datasets in order to predict the distribution of the future demand and model the uncertainty in the context of cloud workload prediction. We compare them to the same models trained with one single dataset, showing that more data help the model to generalize and make it more accurate. Moreover, we provide an ablation study of a bivariate model trained with multiple datasets and compare the results to the univariate counterpart to investigate which components are more affecting the accuracy of the prediction and its impact on the QoS.

Results show that... \\

Contextually, we make available the 12 datasets from the Google Cloud and Alibaba Cloud Computing systems' traces that we pre-processed and used for the experiments to facilitate the research on this field. 
