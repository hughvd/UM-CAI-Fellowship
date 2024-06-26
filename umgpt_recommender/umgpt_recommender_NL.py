import pandas as pd
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from typing import List, Optional




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
        print('Initializing...')
        self.df = df

        #Sets the current working directory to be the same as the file.
        os.chdir(os.path.dirname(os.path.abspath('umgpt_recommender.py')))

        #Load environment file for secrets, set to .env path.
        try:
            if load_dotenv(r'C:\Users\hvand\OneDrive - Umich\Documents\atlas\umgpt_recommender\.env') is False:
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

    def recommend(self, levels: Optional[List[int]] = None, query: str = ''):
        print('Recommending...')
        system_content = '''
        You are a keyword extraction tool used by a College Course Recommendation System that searches through course descriptions to recommend classes to a student.
        You will output a series of keywords in the specified format based on a students request to help the system filter the dataset to relevant courses. 
        Example:
        Student request: "I am a mathematics student interested in computer science theory. What are some courses I could take?"
        Your output: "computer science, algorithms, theory, data structures, discrete mathematics, computation, computational complexity"
        
        '''
        messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": query}
            ]
        print('Initial filter')
        gpt_response = self.client.chat.completions.create(
            model=os.environ['OPENAI_MODEL'],
            messages=messages,
            temperature=0,
            stop=None)
        messages.append({"role": "system", "content": gpt_response.choices[0].message.content})
        keywords = gpt_response.choices[0].message.content.split(',')
        keywords = [word.strip().lower() for word in keywords]

        print('Initial keywords:')
        print(keywords)

        if levels is None:
            levels = []
        
        if levels:
            filtered_df = self.df[self.df['level'].isin(levels)]
            filtered_df = filtered_df[filtered_df['description'].str.contains('|'.join(keywords), case=False, na=False)]
        else:
            filtered_df = self.df[self.df['description'].str.contains('|'.join(keywords), case=False, na=False)]
        print(f"Initial size: {filtered_df.shape[0]}")

        ## Turn the remaining courses into one long string.
        course_string = ''
        for _, row in filtered_df.iterrows():
            course_name = row['course']
            description = row['description']
            course_string += f"Course {course_name}: {description}\n"

        system_rec_message = "You are the worlds most highly trained academic advisor, a student has come to you with the following request: \n"
        system_rec_message = system_rec_message + query + '\n'
        system_rec_message = system_rec_message + '''Recommend the best courses from the following list, 
                                                    return your recommendations as a list of the courses and a short rationale:\n''' + course_string
        recommendation = self.client.chat.completions.create(
            model=os.environ['OPENAI_MODEL'],
            messages=[
                {'role': 'system', 'content': system_rec_message}
                ],
            temperature=0,
            stop=None)
        
        print('Returning...')
        print(recommendation.choices[0].message.content)
        return recommendation.choices[0].message.content
    


class EmbeddingRecommender(object):
    '''
    This recommender uses embeddings to find courses most similar to a given students request.
    '''
    def __init__(self, df: pd.DataFrame):
        """Initialize course recommender to given Pandas dataframe. Dataframe must have 
        columns labeled as ['course', 'description', 'embedding']. Loads OpenAI gpt and embedding model, must have .env file with 
        'OPENAI_API_KEY' = your_api_key.
        """
        super().__init__()
        print('Initializing...')
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