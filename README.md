# Nationality prediction

Several models are available online for predicting nationality from names, but they are not without drawbacks, for example:
* [nationalize.io](https://nationalize.io/): Free access is limited and may not meet larger-scale needs.
* [NamePrism](https://www.name-prism.com/): Requires applying through a form and getting accepted for API tokens, making the process less accessible.
* [name2nat](https://github.com/Kyubyong/name2nat): An open-source option, but outdated and difficult to set up due to unclear dependencies.

This project offers a fresh alternative:

* Fully free: No subscription or usage caps.
* API-free and offline: A model you can run locally without internet access.
* Easy to set up and integrate, with all dependencies clearly defined.

## Usage

```python
from natpred import NationalityPrediction

nat_pred = NationalityPrediction()

prediction = nat_pred.predict('Antonino Pizzolato', 7)
print(prediction)
```

Output:
```
{'IT': 0.9919720888137817,
 'AT': 0.004203251097351313,
 'ES': 0.0007780595915392041,
 'HU': 0.0004789325757883489,
 'RU': 0.0004524968971963972,
 'FR': 0.00042069185292348266,
 'BE': 0.0003427563642617315}
```

## Details

### Setting up the Conda environment

To set up the Conda environment, run the following command in your terminal:
```commandline
conda env create -f dependencies.yml
```
This will create a new environment with the required dependencies. Activate it using:

```commandline
conda activate nat_pred
```

### Dataset and Training
To download the dataset, simply run:
```bash
python data_downloader.py  
```

To train the model using the downloaded dataset, use the following command:
```bash
python training.py 
```



### Data

Freely available datasets for nationality prediction are extremely limited, with most relying on artificially generated data. To overcome this limitation, the dataset for this project was created entirely from scratch using publicly available information on Wikipedia.

The data was gathered using **Wikipedia's API**, allowing for efficient and accurate extraction of relevant information. Special care was taken to focus on categories that are explicitly linked to nationalities, such as: https://en.wikipedia.org/wiki/Category:21st-century_English_people. This method ensures that the dataset reflects real-world examples and provides a strong foundation for building the prediction model.

Currently, the model supports the following European countries:
Austria (AT), Belarus (BY), Belgium (BE), Bulgaria (BG), Czech Republic (CZ), Denmark (DK), Finland (FI), France (FR), Germany (DE), Greece (GK), Hungary (HU), Ireland (IE), Italy (IT), Netherlands (NL), Norway (NO), Poland (PL), Portugal (PT), Romania (RO), Russia (RU), Slovakia (SK), Spain (ES), Sweden (SE), Ukraine (UA), and the United Kingdom (GB).

### Model and hyperparameters

To achieve accurate nationality predictions, several sequence-based machine learning models were tested. Among them, a **GRU** (Gated Recurrent Unit) model demonstrated the best performance (**64.3%**), effectively capturing the patterns and dependencies in the name sequences. GRUs were particularly well-suited for this task due to their ability to handle sequential data efficiently while being less complex than other recurrent architectures like LSTMs.

The optimal hyperparameters for the model were determined using [**Hyperband**](https://arxiv.org/abs/1603.06560), a cutting-edge bandit-based algorithm for hyperparameter optimization. Hyperband is particularly suitable for scenarios with limited computational resources, such as running experiments locally on a personal computer. Compared to traditional methods like grid search or random search, Hyperband achieves faster convergence by dynamically allocating resources to the most promising configurations, making it both resource-efficient and effective for this project.

## Next steps
* **!!! Improve the codebase**: Refactoring and removing hard-coded elements to make the code more flexible, modular, and maintainable for future improvements and scalability.
* **Test and compare more models, methods, and algorithms** such as Supervised Contrastive Learning and LLM prompting to further improve prediction accuracy and robustness. 
* **Train models on a larger dataset**.
* **Expand the dataset to cover more countries**.
* **Publish the package on PyPI**: Making the project easily installable via pip to increase accessibility for users and developers.
* **Analyze how well the model performs on names from countries with linguistically or culturally similar populations**, such as Russia, Ukraine, and Belarus.
