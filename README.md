# Poisonous Mushroom Classifier
- author: Amar Gill, Ruth Yankson, Limor Winter & Claire Saunders
## About
In this project, we built a supervised classification model using a Support Vector Classifier (SVC) to predict whether a mushroom from the Agaricus and Lepiota family is poisonous or edible based on its physical characteristics. The model uses categorical features such as odor, gill color, stalk shape, and habitat to distinguish between the two classes. Our goal was to explore how effectively a machine learning model can identify poisonous mushrooms and thereby reduce the risk of mushroom poisoning caused by misidentification.
Our final SVC model demonstrated perfect predictive performance on the unseen test set, achieving precision, recall, and overall accuracy of 1.0. These results align with the benchmark performance reported by the UCI Machine Learning Repository, confirming the dataset’s suitability for classification tasks. While the model’s accuracy is impressive, its reliability in real-world settings would depend heavily on the user’s ability to correctly identify each mushroom’s physical features.

The dataset used was adapted from The Audubon Society Field Guide to North American Mushrooms by Gary Lincoff (1981) and contains 8124 hypothetical samples representing 23 mushroom species. Each sample includes 22 categorical features describing observable physical traits and a binary label indicating whether the mushroom is edible or poisonous. Future work could focus on testing the model with real-world image and exploring feature simplification to improve accessibility and practical use by amateur foragers.

## Project Structure
- `notebooks/poisonous_mushroom_classifier.qmd`: Quarto notebook containing the full analysis and report.
The repository is organized as follows:

- `data/`
  - `raw/`: Contains the original downloaded dataset.
  - `processed/`: Contains any intermediate processed data files.
- `notebooks/`
  - `poisonous_mushroom_classifier.qmd`: Main analysis document.
  - `poisonous_mushroom_classifier.pdf`: Rendered report.
- `environment.yml`: Conda environment file listing the required packages.
- `conda-lock.yml`: Conda lock file specifying exact package versions for reproducibility.
- `CODE_OF_CONDUCT.md`: Community guidelines for respectful collaboration.
- `CONTRIBUTING.md`: Instructions for how others can contribute to the project.
- `LICENSE`: License for the code and documentation.
- `README.md`: Project overview and instructions.
- `.gitignore`: Specifies which files and folders Git should ignore.

## Report
The final report can be found [here](notebooks/poisonous_mushroom_classifier.qmd)
## Usage
First time running the project, run the following from the root of this repository:

```bash
conda-lock install --name poisonous_mushroom_classifier conda-lock.yml
```

Then, activate the environment:

```bash
conda activate poisonous_mushroom_classifier
```

To run the analysis, run the following from the root of this repository:

```bash
quarto render notebooks/poisonous_mushroom_classifier.qmd
```

## Dependencies
- conda (version 23.9.0 or higher)
- conda-lock (version 2.5.7 or higher)
- jupyterlab (version 4.0.0 or higher)
- nb_conda_kernels (version 2.3.1 or higher)
- Quarto** (version 1.4.0 or higher)
- Python and packages listed in environment.yml

## License
The Poisonous Mushroom Classifier report and documentation contained herein are licensed under the **Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** License.  
See the [LICENSE](LICENSE) file for more information.

If re-using or re-mixing, please provide attribution and a link to this repository.

The software code contained within this repository is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for more information.

## References
Lincoff, G. H. (1981).The Audubon Society Field Guide to North American Mushrooms (1981). New York: Alfred A. Knopf

Mushroom. (1981). UCI Machine Learning Repository. https://doi.org/10.24432/C5959T

Diaz, J. H. (2016). Mistaken mushroom poisonings. Wilderness & Environmental Medicine, 27(2), 330-335

Buitinck, L., Louppe, G., Blondel, M., Pedregosa,               Fabian, Mueller, A., Grisel, O., … Ga"el Varoquaux. (2013). API design for machine learning software: experiences from the scikit-learn              project. In ECML PKDD Workshop: Languages for Data Mining and Machine Learning (pp. 108–122).
