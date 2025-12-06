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
# Setup
If you are using Windows or Mac, make sure Docker Desktop is running.

1. Clone this GitHub repository.

# Running the analysis

1. Navigate to the root of this project on your computer using the command line and enter the following command:

```bash
docker compose run --service-ports --remove-orphans analysis-env
```
2. In the terminal, look for a URL that starts with http://127.0.0.1:8888/lab?token=... (for an example, see the highlighted text in the terminal below). Copy and paste that URL into your browser. This will open Jupyter Lab inside the Docker container.

3. To run the full analysis and render the report, open a terminal inside Jupyter Lab and run:

```bash
quarto render docs/poisonous_mushroom_classifier.qmd
```

This command will reproduce the full analysis and generate the report file:

```bash
docs/index.html
```
You can then view the rendered report in your browser by opening docs/index.html.

# Clean up
1. To shut down the container and clean up the resources, type `Cntrl + C` in the terminal where you launched the container, and then type `docker compose rm`

## Developer notes

# Developer dependencies
- `conda` (version 23.9.0 or higher)
- `conda-lock` (version 2.5.7 or higher)

# Adding a new dependency
To add a new Python or system dependency:
1. Add the dependency to the `environment.yml` file on a new branch.
2. Run `conda-lock -k explicit --file environment.yml -p linux-64` to update the `conda-linux-64.lock` file.
3. Re-build the Docker image locally to ensure it builds and runs properly.
4. Push the changes to GitHub. A new Docker image will be built and pushed to Docker Hub automatically.
5. Update the `docker-compose.yml` file on your branch to use the new container image (make sure to update the tag specifically).
6. Send a pull request to merge the changes into the `main` branch.

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
