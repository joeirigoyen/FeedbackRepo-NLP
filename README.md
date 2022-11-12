# FeedbackRepo-NLP
This repository contains three tasks that represents what it is to be a NLP engineer in a real working environment. 

# How to set API keys
In order to perform language translation, the following engines are used within the project:

    Google Cloud
    DeepL

To ensure that DeepL performs correctly, the following variable must be set in an ```.env``` file in the project's root directory:

    DEEPL_API_KEY=<API_KEY_STRING>

To ensure that Google Cloud performs correctly, the platform's generated JSON file must be stored in the project's root directory as ```cloudapikey.json```.


# How to run
For the flow tester to be ran, it is necessary to execute the following command at the root directory of the project (```/FeedbackRepo-NLP```):

    pip install -r requirements.txt && python run.py