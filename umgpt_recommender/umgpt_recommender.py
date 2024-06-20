import numpy as np
import pandas as pd
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from typing import List, Optional
import heapq



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

        #Load environment file for secrets.
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

        num_cycles = 0
        print('Begin filter loop')
        while filtered_df.shape[0] > 500 and num_cycles < 2:
            print('Filtering...')
            messages.append({"role": "user", "content": 'Return additional keywords not listed above to filter dataframe.'})
            gpt_response = self.client.chat.completions.create(
            model=os.environ['OPENAI_MODEL'],
            messages=messages,
            temperature=0,
            stop=None)

            new_keywords = gpt_response.choices[0].message.content

            print(f'Keywords {0}:')
            print(new_keywords)

            messages.append({"role": "system", "content": new_keywords})
            # Get keywords in usable form
            new_keywords = new_keywords.split(',')
            new_keywords = [word.strip().lower() for word in new_keywords]
            ## Check that filtered df size is nonzero
            test_df = filtered_df[filtered_df['description'].str.contains('|'.join(new_keywords), case=False, na=False)]
            if len(test_df) > 15:
                filtered_df = test_df
            num_cycles += 1
            print(f"Cycle {num_cycles}: {filtered_df.shape[0]}")
        
        print(f'Final df size: {filtered_df.shape[0]}')
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
        #
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
    # Notes on optimizing:
    # Currently using the pandas array to access each embedding vector, to compute most similar courses 
    # by saving it as a matrix and getting row index of highest similarity would be more efficient.
    # TODO:
    # Metadata filtering to allow for more precise filtering?
    def recommend(self, levels: Optional[List[int]] = None, query: str = ''):
        print('Recommending...')
        system_content = '''
        Return an example course description of a course that would be most applicable to the following students request.
        Student request: 
        '''
        system_content = system_content + query
        messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": query}
            ]
        print('Example description')
        gpt_response = self.client.chat.completions.create(
            model=os.environ['OPENAI_MODEL'],
            messages=messages,
            temperature=0,
            stop=None).choices[0].message.content
        print(gpt_response)
        if levels is None:
            levels = []
        
        if levels:
            filtered_df = self.df[self.df['level'].isin(levels)]
        else:
            filtered_df = self.df
        print(f"Initial size: {filtered_df.shape[0]}")
        ## Turn the remaining courses into one long string.

        ex_embedding = self.client.embeddings.create(
            input = [gpt_response], 
            model=os.environ['OPENAI_EMBEDDING_MODEL']).data[0].embedding
        
        # Get the top 100 similar courses.
        # Loop through dataframe and keeping a stack of top 100 similar courses
        # research running average algorithms
        heap = []
        for idx, row in filtered_df.iterrows():
            # Get cosine similarity of i-th course description and example course.
            similarity = cosine_similarity(ex_embedding, row['embedding'])
            if idx < 100:
                heapq.heappush(heap, (similarity, idx))
            else:
                heapq.heappushpop(heap, (similarity, idx))
        
        # Extract indexes
        indexes = [idx for sim, idx in heap]

        # If you want the actual vectors or rows, you can use:
        filtered_df = filtered_df.iloc[indexes]
        
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
        return recommendation.choices[0].message.content

# HELPER FUNCTIONS
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)