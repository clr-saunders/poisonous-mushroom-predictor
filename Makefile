# Authors: Amar Gill
# Date: Dec-12-2025

.PHONY: all download preprocess eda train report clean

# Default target: Running 'make' will run the whole pipeline
all: report

# 1. Download Data
download:
	python scripts/download_data.py --url "https://archive.ics.uci.edu/static/public/73/mushroom.zip" --data_path "data/raw"

# 2. Split and Preprocess
preprocess: download
	PYTHONPATH=. python scripts/split_n_preprocess.py --raw-data "data/raw/agaricus-lepiota.data"

# 3. EDA 
eda: preprocess
	PYTHONPATH=. python scripts/eda.py

# 4. Model
train: preprocess
	PYTHONPATH=. python scripts/model.py "data/processed/mushroom_train.csv" "data/processed/mushroom_test.csv"

# 5. Render Report
report: eda train
	quarto render docs/poisonous_mushroom_classifier.qmd

# Clean up generated files
clean:
	rm -rf data/raw data/processed
	rm -f docs/*.html docs/*.pdf
	rm -rf docs/poisonous_mushroom_classifier_files
	rm -rf results
	rm -rf scripts/models
