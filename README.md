# Poisonous Mushroom Classifier
- author: Amar Gill, Ruth Yankson, Limor Winter & Claire Saunders
## About
In this project, we built a supervised classification model using a Support Vector Classifier (SVC) to predict whether a mushroom from the Agaricus and Lepiota family is poisonous or edible based on its physical characteristics. The model uses categorical features such as odor, gill color, stalk shape, and habitat to distinguish between the two classes. Our goal was to explore how effectively a machine learning model can identify poisonous mushrooms and thereby reduce the risk of mushroom poisoning caused by misidentification.
Our final SVC model demonstrated perfect predictive performance on the unseen test set, achieving precision, recall, and overall accuracy of 1.0. These results align with the benchmark performance reported by the UCI Machine Learning Repository, confirming the dataset’s suitability for classification tasks. While the model’s accuracy is impressive, its reliability in real-world settings would depend heavily on the user’s ability to correctly identify each mushroom’s physical features.

The dataset used was adapted from The Audubon Society Field Guide to North American Mushrooms by Gary Lincoff (1981). It was sourced from the UCI Machine Learning Repository (Dua and Graff 2017) and can be found [here](https://archive.ics.uci.edu/dataset/73/mushroom), specifically this file: *agaricus-lepiota.data*. Each row in the data set represents one mushroom sample, including 22 categorical features describing observable traits and a binary label indicating whether the mushroom is edible or poisonous. Future work could focus on testing the model with real-world image and exploring feature simplification to improve accessibility and practical use by amateur foragers.

## Report
The final report can be found [here](docs/poisonous_mushroom_classifier.qmd)

## Dependencies
- [Docker](https://www.docker.com/)
- [VS Code](https://https://code.visualstudio.com/download)
- [VS Code Jupyter Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

## Usage
First time running the project, run the following from the root of this repository:

Estimate time: 60-120s (Dependent on network speed, CPU and storage media)

```bash
docker compose run --service-ports --remove-orphans analysis-env
```


To run the analysis inside the container, run the following within:

```bash
quarto render docs/poisonous_mushroom_classifier.qmd
```



## License
The Poisonous Mushroom Classifier report and documentation contained herein are licensed under the **Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** License.  
See the [LICENSE](LICENSE) file for more information.

If re-using or re-mixing, please provide attribution and a link to this repository.

The software code contained within this repository is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for more information.

## References
Lincoff, G. H. (1981).The Audubon Society Field Guide to North American Mushrooms (1981). New York: Alfred A. Knopf.

Buitinck, L., Louppe, G., Blondel, M., Pedregosa,               Fabian, Mueller, A., Grisel, O., … Ga"el Varoquaux. (2013). API design for machine learning software: experiences from the scikit-learn              project. In ECML PKDD Workshop: Languages for Data Mining and Machine Learning (pp. 108–122).

Mushroom. (1981). UCI Machine Learning Repository. https://doi.org/10.24432/C5959T.
