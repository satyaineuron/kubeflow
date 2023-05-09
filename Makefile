SHELL := /bin/bash # Use bash syntax

data_ingestion:
	bash scripts/build_and_push_component.sh imdb/data_ingestion imdb_data_ingestion

model_training:
	bash scripts/build_and_push_component.sh imdb/model_training imdb_model_training

build_components: data_ingestion model_training