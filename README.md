# stack-overflow-answer-generator

# Requirements

Python 3.7
[Google BigQuery api credentials](https://console.cloud.google.com/projectselector2/apis/credentials) (as json)

# Installation

1. ```pip install pipenv```
2. Navigate to repository in command prompt (directory should contain pipfile)
3. ```pipenv install```
4. ```pipenv shell```
5. Download checkpoints and tokenizers
6. Extract into repository folder
7. In validate.py change GOOGLE_APPLICATION_CREDENTIALS path to your key path
8. To verify installation run validate.py in pipenv shell
