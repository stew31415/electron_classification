# electron_classification
Code used to classify simulated single and double electron events generated using MaGe for MJD.

NOTE: The simulated single and double electron event datasets were too large to push to GitHub, so they are not present here. The datasets can be obtained by messaging the author at alexstewart314@gmail.com. These simulated data sets were made using MJD/LEGEND's MaGe software in a PPC detector volume.

pytorch_classify: CNN built with PyTorch in order to classify the simulated single and double electron events. Plots ROC curve and calculates AUC for the network.

electron_optimize: Hyperparameter search using AX to optimize the CNN used in pytorch_classify.
