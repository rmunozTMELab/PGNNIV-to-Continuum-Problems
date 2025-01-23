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
