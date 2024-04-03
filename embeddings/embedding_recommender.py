import pandas as pd
import openai as OpenAI
import os
from dotenv import load_dotenv




class EmbeddingRecommender(object):
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
        os.chdir(os.path.dirname(os.path.abspath('embedding_recommender.py')))

        #Load environment file for secrets.
        try:
            if load_dotenv('.env') is False:
                raise TypeError
        except TypeError:
            print('Unable to load .env file.')
            quit()
        #Create Azure client
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
        )

    def _get_embedding(self, text, model="text-embedding-3-small"):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input = [text], model=model).data[0].embedding

    def recommend(self, query: str):
        query_embedding = self._get_embedding(query)
        