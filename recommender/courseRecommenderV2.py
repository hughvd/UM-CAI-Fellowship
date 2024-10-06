import numpy as np
import pandas as pd
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from typing import List, Optional
import heapq
import time

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
        os.chdir(os.path.dirname(os.path.abspath('umgptRecommender.py')))

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
        print('Success')
    # Notes on optimizing:
    # Currently using the pandas array to access each embedding vector, to compute most similar courses 
    # by saving it as a matrix and getting row index of highest similarity would be more efficient.
    # TODO:
    # Metadata filtering to allow for more precise filtering?
    def recommend(self, levels: Optional[List[int]] = None, query: str = '', debug: bool = False):
        print('Recommending...')
        #### Different prompts ####################
        system_content = f"""You will be given a request from a student at The University of Michigan to provide quality course recommendations. \
Generate a course description that would be most applicable to their request. In the course description, provide a list of topics as well as a \
general description of the course. Limit the description to be less than 200 words.

Student Request:
{query}
"""
#         system_content = '''
# You will be given a request from a student at The University of Michigan to provide quality course recommendations. \
# You will return a course description that would be most applicable to their request. In this course descriptions, \
# provide a list of topics as well as a general description of the course. Limit the  description to be less than \
# 500 words.'''
        
        # system_content = '''
        # Return an example course description of a course that would be most applicable to the following students request.
        # Student request: 
        # '''
        messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": query}
            ]
        if debug:
            print('Example description')
        # Generate example description based off queuery.
        tic = time.perf_counter()
        gpt_response = self.client.chat.completions.create(
            model=os.environ['GENERATOR_MODEL'],
            messages=messages,
            temperature=0,
            stop=None).choices[0].message.content
        toc = time.perf_counter()
        timeExDesc = toc - tic
        if debug:
            print(gpt_response)

        # Filter dataframe by course levels
        if levels is None:
            levels = []
        
        if levels:
            # Make sure to reset indices for search later
            filtered_df = self.df[self.df['level'].isin(levels)].reset_index(drop=True)
        else:
            filtered_df = self.df
        # Generate the embedding for example course description
        tic = time.perf_counter()
        ex_embedding = self.client.embeddings.create(
            input = [gpt_response], 
            model=os.environ['OPENAI_EMBEDDING_MODEL']).data[0].embedding
        toc = time.perf_counter()
        timeEmb = toc - tic

        # Get the top 100 similar courses 
        tic = time.perf_counter()
        heap = []
        for idx, row in filtered_df.iterrows():
            similarity = cosine_similarity(ex_embedding, row['embedding'])
            if idx < 100:
                heapq.heappush(heap, (similarity, idx))
            else:
                heapq.heappushpop(heap, (similarity, idx))
        toc = time.perf_counter()
        timeGenHeap = toc - tic
        # Extract indexes and filter
        indexes = [idx for sim, idx in heap]
        filtered_df = filtered_df.iloc[indexes]
        
        # Prepare the courses to be passed into LLM
        course_string = ''
        for _, row in filtered_df.iterrows():
            course_name = row['course']
            description = row['description']
            course_string += f"{course_name}: {description}\n"
        
        system_rec_message = f"""You are the world's most highly trained academic advisor, with decades of experience \
in guiding students towards their optimal academic paths. Your task is to provide personalized course recommendations \
based on the student's profile:

Instructions:
1. Analyze the student's profile carefully, considering their interests, academic background, and career goals.
2. Review the list of available courses provided below.
3. Recommend the top 5-10 most suitable courses for this student.
4. For each recommended course, provide a brief but compelling rationale (1-3 sentences) explaining why it's a good fit.
5. Format your response as a numbered list, with each item containing the course name followed by your rationale.

Student Profile:
{query}

Available Courses:
{course_string}

Remember: Your recommendations should be tailored to the student's unique profile and aspirations. Aim to balance academic growth, career preparation,\
and personal interest in your selections."""
        

        # Recommend with streaming
        tic = time.perf_counter()
        stream = self.client.chat.completions.create(
            model=os.environ['RECOMMENDER_MODEL'],
            messages=[
                {'role': 'system', 'content': system_rec_message}
            ],
            temperature=0,
            stream=True
        )

        # Output the stream
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end='', flush=True) 
                full_response += content

        toc = time.perf_counter()
        timeRec = toc - tic

        if debug:
            print('Runtime Info: ')
            print(f'Time to generate example description: {timeExDesc:.2f}')
            print(f'Time to generate embedding: {timeEmb:.2f}')
            print(f'Time to generate heap: {timeGenHeap:.2f}')
            print(f'Time to generate recommendation: {timeRec:.2f}')
        print('Returning...')
        return full_response 


################################### END OF CLASS ###################################################
# HELPER FUNCTIONS
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)