#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Download the model.keras file
wget https://raw.githubusercontent.com/shrenisc/Translator/main/language_translation_model.keras -O language_translation_model.keras

# Decompress the model file
gunzip language_translation_model.keras.gz
