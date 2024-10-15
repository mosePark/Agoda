#%%

import os
import numpy as np
import pandas as pd
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

#%%
'''
API
'''

current_directory = os.getcwd()
project_root = current_directory


env_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=env_path)


claude_api_key = os.getenv("CLAUDE_API_KEY")

#%%

#%%
'''
DATA
'''

# os.chdir('C:/Users/mose/agoda/data/') # home
# os.chdir('C:/Users/UOS/Desktop/Agoda-Data/raw') # lab
os.chdir('D:/Agoda-Data/raw') # Lab D-drive


df = pd.read_csv("agoda2.csv")

#%%

os.chdir('D:/Agoda-Data/claude')

#%%

'''
Prompt
'''

system_template = """
Read the following review text and generate another review text with a similar context:

Note: The original text is composed in the form of title + "\n" + body, as shown below:
"""


user_template = "Original Review: {Full}\n\nPlease generate a review in a similar format, with a title and a body:\n\nGenerated Review:"


prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", user_template)
])


llm = ChatAnthropic(api_key=claude_api_key, model_name="claude-2.1")


parser = StrOutputParser()


chain = llm | parser

#%%

original_review = df.loc[0, "Full"]


messages = prompt_template.invoke({"Full": original_review})

try:
    generated_review = chain.invoke(messages)

    if isinstance(generated_review, str):
        print("Generated Review:", generated_review.strip())
    else:
        print("Generated Review could not be parsed correctly.")
except Exception as e:

    print("Error generating review: " + str(e))
