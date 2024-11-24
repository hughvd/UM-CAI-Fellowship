import numpy as np
import pandas as pd
from openai import AsyncAzureOpenAI
from typing import List, Optional, Dict, Any, AsyncGenerator
import heapq
from abc import ABC, abstractmethod
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncOpenAIClient:
    def __init__(self, config: Dict[str, str]):
        self.client = AsyncAzureOpenAI(
            api_key=config["OPENAI_API_KEY"],
            api_version=config["OPENAI_API_VERSION"],
            azure_endpoint=config["OPENAI_API_BASE"],
            organization=config["OPENAI_ORGANIZATION_ID"]
        )
        self.generator_model = config["GENERATOR_MODEL"]
        self.rec_model = config["RECOMMENDER_MODEL"]
        self.embedding_model = config["OPENAI_EMBEDDING_MODEL"]

    async def generate_chat_completion(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        try:
            response = await self.client.chat.completions.create(
                model=self.rec_model,
                messages=messages,
                temperature=0,
                stream=True
            )
            async for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Error in generate_chat_completion: {str(e)}")
            raise

    async def generate_embedding(self, text: str) -> List[float]:
        try:
            response = await self.client.embeddings.create(
                input=[text],
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error in generate_embedding: {str(e)}")
            raise

class SimilarityCalculator(ABC):
    @abstractmethod
    def calculate(self, vec1: List[float], vec2: List[float]) -> float:
        pass

class CosineSimilarityCalculator(SimilarityCalculator):
    def calculate(self, vec1: List[float], vec2: List[float]) -> float:
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

class EmbeddingRecommender:
    def __init__(self, openai_client: AsyncOpenAIClient, similarity_calculator: SimilarityCalculator):
        self.openai_client = openai_client
        self.similarity_calculator = similarity_calculator
        self.courses_df: Optional[pd.DataFrame] = None

    def load_courses(self, courses_data: List[Dict[str, Any]]):
        self.courses_df = pd.DataFrame(courses_data)

    async def generate_example_description(self, query: str) -> str:
        system_content = f"""You will be given a request from a student at The University of Michigan to provide quality course recommendations. \
Generate a course description that would be most applicable to their request. In the course description, provide a list of topics as well as a \
general description of the course. Limit the description to be less than 200 words.

Student Request:
{query}
"""
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ]
        try:
            response = await self.openai_client.client.chat.completions.create(
                model=self.openai_client.generator_model,
                messages=messages,
                temperature=0,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating example description: {str(e)}")
            raise

    def find_similar_courses(self, filtered_df: pd.DataFrame, example_embedding: List[float], top_n: int = 50) -> List[int]:
        heap = []
        for idx, row in filtered_df.iterrows():
            similarity = self.similarity_calculator.calculate(example_embedding, row['embedding'])
            if len(heap) < top_n:
                heapq.heappush(heap, (similarity, idx))
            elif similarity > heap[0][0]:
                heapq.heappushpop(heap, (similarity, idx))
        return [idx for _, idx in heap]

    async def stream_recommend(self, query: str, levels: Optional[List[int]] = None) -> AsyncGenerator[str, None]:
        try:
            if self.courses_df is None:
                raise ValueError("Courses have not been loaded. Call load_courses() first.")

            # Generate example description and its embedding
            try:
                example_description = await self.generate_example_description(query)
            except Exception as e:
                yield f"Error generating example description: {str(e)}"
                return

            try:
                example_embedding = await self.openai_client.generate_embedding(example_description)
            except Exception as e:
                yield f"Error generating embedding: {str(e)}"
                return

            # Filter courses by level and similarity
            filtered_df = self.courses_df if levels is None else self.courses_df[self.courses_df['level'].isin(levels)]
            # Reset the index of filtered_df
            filtered_df = filtered_df.reset_index(drop=True)
            
            similar_course_indices = self.find_similar_courses(filtered_df, example_embedding)
            filtered_df = filtered_df.iloc[similar_course_indices]


            # Prepare course string for the prompt
            course_string = "\n".join(f"{row['course']}: {row['description']}" for _, row in filtered_df.iterrows())

            # Prepare the recommendation prompt
            system_rec_message = f"""You are an expert academic advisor specializing in personalized course recommendations. \
When evaluating matches between student profiles and courses, prioritize direct relevance, prerequisite alignment, and career trajectory fit.

Context: Student Profile ({query})
Course Options: 
{course_string}

REQUIREMENTS:
- Return exactly 10 courses, ranked by relevance and fit
- Recommend ONLY courses listed in Course Options
- For each recommendation include:
  1. Course number
  2. One-sentence explanation focused on student's specific profile/goals
  3. Confidence level (High/Medium/Low)

FORMAT (Markdown):
1. COURSEXXX
Rationale: [One clear sentence explaining fit]
Confidence: [Level]

2. [Next course...]

CONSTRAINTS:
- NO general academic advice
- NO mentions of prerequisites unless explicitly stated in course description
- NO suggestions outside provided course list
- NO mention of being an AI or advisor"""
            
#             system_rec_message = f"""You are the world's most highly trained academic advisor, with decades of experience \
# in guiding students towards their optimal academic paths. Your task is to provide personalized course recommendations \
# based on the student's profile:

# Instructions:
# 1. Analyze the student's profile carefully, considering their interests, academic background, and career goals.
# 2. Review the list of available courses provided below.
# 3. Recommend the top 5-10 most suitable courses for this student.
# 4. For each recommended course, provide a brief but compelling rationale (2-3 sentences) explaining why it's a good fit.
# 5. Format your response as a numbered list, with each item containing the course name followed by your rationale.

# Student Profile:
# {query}

# Available Courses:
# {course_string}

# Remember: Your recommendations should be tailored to the student's unique profile and aspirations. Aim to balance academic growth, career preparation, \
# and personal interest in your selections. Do not recommend courses that are not under available courses."""

            messages = [{'role': 'system', 'content': system_rec_message}]

            # Stream the recommendation
            try:
                async for token in self.openai_client.generate_chat_completion(messages):
                    yield token
            except Exception as e:
                yield f"Error generating recommendation: {str(e)}"
                
            except Exception as e:
                yield f"Error generating recommendation: {str(e)}"

        except Exception as e:
            yield f"Unexpected error: {str(e)}"

    # Temporary function until UM ITS releases streaming for UMGPT API
    async def recommend(self, query: str, levels: Optional[List[int]] = None) -> str:
        try:
            if self.courses_df is None:
                raise ValueError("Courses have not been loaded. Call load_courses() first.")

            # Generate example description and its embedding
            example_description = await self.generate_example_description(query)
            example_embedding = await self.openai_client.generate_embedding(example_description)

            # Filter courses by level and similarity
            filtered_df = self.courses_df if levels is None else self.courses_df[self.courses_df['level'].isin(levels)]
            filtered_df = filtered_df.reset_index(drop=True)
            
            similar_course_indices = self.find_similar_courses(filtered_df, example_embedding)
            filtered_df = filtered_df.iloc[similar_course_indices]

            # Prepare course string for the prompt
            course_string = "\n".join(f"{row['course']}: {row['description']}" for _, row in filtered_df.iterrows())

            # Prepare the recommendation prompt
            # NOTE: our dataframe does not have the course name as a data column
            system_rec_message = f"""You are an expert academic advisor specializing in personalized course recommendations. \
When evaluating matches between student profiles and courses, prioritize direct relevance, prerequisite alignment, and career trajectory fit.

Context: Student Profile ({query})
Course Options: 
{course_string}

REQUIREMENTS:
- Return exactly 10 courses, ranked by relevance and fit
- Recommend ONLY courses listed in Course Options
- For each recommendation include:
  1. Course number
  2. One-sentence explanation focused on student's specific profile/goals
  3. Confidence level (High/Medium/Low)

FORMAT (Markdown):
1. **COURSEXXX: Couse Name**
Rationale: [One clear sentence explaining fit]
Confidence: [Level]

2. [Next course...]

CONSTRAINTS:
- NO general academic advice
- NO mentions of prerequisites unless explicitly stated in course description
- NO suggestions outside provided course list
- NO mention of being an AI or advisor"""
            
#             system_rec_message = f"""You are the world's most highly trained academic advisor, with decades of experience \
# in guiding students towards their optimal academic paths. Your task is to provide personalized course recommendations \
# based on the student's profile:

# Instructions:
# 1. Analyze the student's profile carefully, considering their interests, academic background, and career goals.
# 2. Review the list of available courses provided below.
# 3. Recommend the top 5-10 most suitable courses for this student.
# 4. For each recommended course, provide a brief but compelling rationale (2-3 sentences) explaining why it's a good fit.
# 5. Format your response as a numbered list, with each item containing the course name followed by your rationale.

# Student Profile:
# {query}

# Available Courses:
# {course_string}

# Remember: Your recommendations should be tailored to the student's unique profile and aspirations. Aim to balance academic growth, career preparation, \
# and personal interest in your selections. Do not recommend courses that are not under available courses."""

            messages = [{'role': 'system', 'content': system_rec_message}]

            response = await self.openai_client.client.chat.completions.create(
                model=self.openai_client.rec_model,
                messages=messages,
                temperature=0,
                stream=False
            )
            recommendation = response.choices[0].message.content
            return recommendation
        except Exception as e:
            return f"Error: {str(e)}"

# Usage example:
# config = {
#     "OPENAI_API_KEY": "your_api_key",
#     "OPENAI_API_VERSION": "your_api_version",
#     "OPENAI_API_BASE": "your_api_base",
#     "OPENAI_ORGANIZATION_ID": "your_org_id",
#     "GENERATOR_MODEL": "your_generator_model",
#     "RECOMMENDER_MODEL": "your_recommender_model",
#     "OPENAI_EMBEDDING_MODEL": "your_embedding_model"
# }
# openai_client = AsyncOpenAIClient(config)
# similarity_calculator = CosineSimilarityCalculator()
# recommender = EmbeddingRecommender(openai_client, similarity_calculator)
# recommender.load_courses(courses_data)
# 
# async def print_recommendation():
#     async for token in recommender.stream_recommend("I'm interested in machine learning and data science."):
#         print(token, end='', flush=True)
# 
# import asyncio
# asyncio.run(print_recommendation())