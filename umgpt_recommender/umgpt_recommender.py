import pandas as pd
from openai import AzureOpenAI
import os
from dotenv import load_dotenv




class Recommender(object):
    """Description
    """

    def __init__(self, df: pd.DataFrame):
        """Initialize course recommender to given Pandas dataframe. Dataframe must have 
        columns labeled as ['course', 'description', 'embedding']. Loads OpenAI api, must have .env file with 
        'OPENAI_API_KEY' = your_api_key.
        """
        super().__init__()
        #
        self.df = df

        #Sets the current working directory to be the same as the file.
        os.chdir(os.path.dirname(os.path.abspath('umgpt_recommender.py')))

        #Load environment file for secrets.
        try:
            if load_dotenv('.env') is False:
                raise TypeError
        except TypeError:
            print('Unable to load .env file.')
            quit()
        #Create Azure client
        self.client = AzureOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            api_version=os.environ['OPENAI_API_VERSION'],
            azure_endpoint=os.environ['OPENAI_API_BASE'],
            organization=os.environ['OPENAI_ORGANIZATION_ID']
        )

    def recommend(self, query: str):
        system_content = '''
        You are a keyword extraction tool used by a College Course Recommendation System that searches through course descriptions to recommend classes to a student.
        You will output a series of keywords in the specified format based on a students request to help the system filter the dataset to relevant courses. 
        Example:
        Student request: "I am a mathematics student interested in computer science theory. What are some courses I could take?"
        Your output: "computer science, algorithms, theory, data structures, discrete mathematics, computation, computational complexity"
        
        '''
        gpt_response = self.client.chat.completions.create(
            model=os.environ['OPENAI_MODEL'],
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": query}
            ],
            temperature=0,
            stop=None)
        print(gpt_response.choices[0].message.content)