# Physically-Guided-Neural-Networks-with-Internal-Variables-to-Continuum-Problems
This README provides an overview of the workflow for implementing PGNNIV models to obtain the results of the paper "On the application of Physically-Guided Neural Networks with Internal Variables to Continuum Problems".

### Data Generation
The data generators are located in the ``src\nb\data_generators``. When you run these notebooks, data for specific problems will be generated and saved in the data folder. You can adjust various parameters (e.g., mesh, number of data points) and use example functions obtained with the Manufactured Solutions Method.

### Neural Network Training
After generating the data, proceed to train the neural networks. There are three key files in the ``src\python``, all starting with ‘PGNNIV’, one for each kind of dataset. Additionally, there is an ```executable.py``` file that allows you to run all training programs in one execution.

Running the Training Programs
To run the training programs, use the following command:

```` python program_name.py number_of_iterations training_mode ````

Training Mode Options
- ```0```: Train the model from scratch, without loading pre-existing weights.
- ```1```: Continue training from a set of pre-existing weights. This mode is useful for refining the model with a lower learning rate after an initial training session.

### Results
The results of the training will be saved in the ```results```. This folder will contain:
- The network weights from both training scenarios (mode 0 and mode 1).
- Training results such as predictions, loss terms, etc.

### Results Analysis
Once the results are available, you can run the notebooks to display various graphs and diagnostics of the models. These notebooks are located in the ```src\nb\results_analysis```:
- Individual notebooks for each model, providing specific analyses.
<!-- - A general notebook in which there are a couple of simple functions that show the boxplots of the relative error for each of the variables and then show pictures of the results for different percentiles of the chosen error. -->

### BibTeX Citation
If you cite this work in a scientific publication, we would appreciate using the following citation:

```
@article{MUNOZSIERRA2025105317,
title = {On the application of Physically-Guided Neural Networks with Internal Variables to Continuum Problems},
journal = {Mechanics of Materials},
volume = {205},
pages = {105317},
year = {2025},
issn = {0167-6636},
doi = {https://doi.org/10.1016/j.mechmat.2025.105317},
url = {https://www.sciencedirect.com/science/article/pii/S0167663625000791},
author = {Rubén Muñoz-Sierra and Jacobo Ayensa-Jiménez and Manuel Doblaré},
keywords = {Physically Guided Neural Networks with Internal Variables, Explanatory Artificial Intelligence, Scientific machine learning, Internal state variables, Continuum physics},
abstract = {Predictive physics has been historically based upon the development of mathematical models that describe the evolution of a system under certain external stimuli and constraints. The structure of such mathematical models relies on a set of physical hypotheses that are assumed to be fulfilled by the system within a certain range of environmental conditions. A new perspective is now raising that uses physical knowledge to inform the data prediction capability of Machine Learning tools, coined as Scientific Machine Learning. A particular approach in this context is the use of Physically-Guided Neural Networks with Internal Variables, where universal physical laws are used as constraints to a given neural network, in such a way that some neuron values can be interpreted as internal state variables of the system. This endows the network with unraveling capacity, as well as better predictive properties such as faster convergence, fewer data needs and additional noise filtering. Besides, only observable data are used to train the network, and the internal state equations may be extracted as a result of the training process, so there is no need to make explicit the particular structure of the internal state model, while getting solutions consistent with Physics. We extend here this methodology to continuum physical problems driven by a general set of partial differential equations, showing again its predictive and explanatory capacities when only using measurable values in the training set. Moreover, we show that the mathematical operators developed for image analysis in deep learning approaches can be used in a natural way and extended to consider standard functional operators in continuum Physics, thus establishing a common framework for both. The methodology presented demonstrates its ability to discover the internal constitutive state equation for some problems, including heterogeneous, anisotropic and nonlinear features, while maintaining its predictive ability for the whole dataset coverage, with the cost of a single evaluation. As a consequence, microstructural material properties can be inferred from macroscopic measurement coming from sensors without the need of specific homogeneous test plans neither specimen extraction.}
}
```

