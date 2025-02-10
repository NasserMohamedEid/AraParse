from ibm_watsonx_ai.foundation_models import Model
from langchain_ibm import WatsonxLLM
import os

def get_credentials():
    return {
        "url": "https://eu-de.ml.cloud.ibm.com",
        "apikey": "Jq8P15FmG-lxwhWU0Zm5mGLVkREuDH4mqTCvy6_UHTg1"
    }

model_id = "sdaia/allam-1-13b-instruct"
parameters = {"decoding_method": "greedy", "max_new_tokens": 900, "repetition_penalty": 1}
project_id = "9c0b7793-7781-4139-8205-44ba471b2f82"
space_id = os.getenv("SPACE_ID")

model = Model(model_id=model_id, params=parameters, credentials=get_credentials(), project_id=project_id, space_id=space_id)
watsonx_llm = WatsonxLLM(watsonx_model=model)