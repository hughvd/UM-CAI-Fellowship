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
            organization=config["OPENAI_ORGANIZATION_ID"],
        )
        self.generator_model = config["GENERATOR_MODEL"]
        self.rec_model = config["RECOMMENDER_MODEL"]
        self.embedding_model = config["OPENAI_EMBEDDING_MODEL"]

    async def generate_chat_completion(
        self, messages: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        try:
            response = await self.client.chat.completions.create(
                model=self.rec_model, messages=messages, temperature=0, stream=True
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
                input=[text], model=self.embedding_model
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
    def __init__(
        self,
        openai_client: AsyncOpenAIClient,
        similarity_calculator: SimilarityCalculator,
    ):
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
        # Load input and get system prompt
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]
        try:
            response = await self.openai_client.client.chat.completions.create(
                model=self.openai_client.generator_model,
                messages=messages,
                temperature=0,
                stream=False,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating example description: {str(e)}")
            raise

    def find_similar_courses(
        self, filtered_df: pd.DataFrame, example_embedding: List[float], top_n: int = 50
    ) -> List[int]:
        heap = []
        for idx, row in filtered_df.iterrows():
            similarity = self.similarity_calculator.calculate(
                example_embedding, row["embedding"]
            )
            if len(heap) < top_n:
                heapq.heappush(heap, (similarity, idx))
            elif similarity > heap[0][0]:
                heapq.heappushpop(heap, (similarity, idx))
        return [idx for _, idx in heap]

    async def stream_recommend(
        self, query: str, levels: Optional[List[int]] = None
    ) -> AsyncGenerator[str, None]:
        try:
            if self.courses_df is None:
                raise ValueError(
                    "Courses have not been loaded. Call load_courses() first."
                )

            # Generate example description and its embedding
            try:
                example_description = await self.generate_example_description(query)
            except Exception as e:
                yield f"Error generating example description: {str(e)}"
                return

            try:
                example_embedding = await self.openai_client.generate_embedding(
                    example_description
                )
            except Exception as e:
                yield f"Error generating embedding: {str(e)}"
                return

            # Filter courses by level and similarity
            filtered_df = (
                self.courses_df
                if levels is None
                else self.courses_df[self.courses_df["level"].isin(levels)]
            )
            # Reset the index of filtered_df
            filtered_df = filtered_df.reset_index(drop=True)

            similar_course_indices = self.find_similar_courses(
                filtered_df, example_embedding
            )
            filtered_df = filtered_df.iloc[similar_course_indices]

            # Prepare course string for the prompt
            course_string = "\n".join(
                f"{row['course']}: {row['description']}"
                for _, row in filtered_df.iterrows()
            )

            # Prepare the recommendation prompt
            system_rec_message = f"""You are an expert academic advisor specializing in personalized course recommendations. \
When evaluating matches between student profiles and courses, prioritize direct relevance and career trajectory fit.

Context: Student Profile ({query})
Course Options: 
{course_string}

REQUIREMENTS:
- Return exactly 10 courses, ranked by relevance and fit
- Recommend ONLY courses listed in Course Options
- For each recommendation include:
  1. Course number
  2. Course name
  2. Two-sentence explanation focused on student's specific profile/goals
  3. Confidence level (High/Medium/Low)

FORMAT (Markdown):
1. **COURSEXXX: COURSE_TITLE**
Rationale: [Two clear sentences explaining fit]
Confidence: [Level]

2. [Next course...]

CONSTRAINTS:
- NO general academic advice
- NO mentions of prerequisites unless explicitly stated in course description
- NO suggestions outside provided course list
- NO mention of being an AI or advisor
- **If multiple courses have identical titles and descriptions (cross-listed), recommend only ONE of them**"""

            messages = [{"role": "system", "content": system_rec_message}]

            # Stream the recommendation
            try:
                async for token in self.openai_client.generate_chat_completion(
                    messages
                ):
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
                raise ValueError(
                    "Courses have not been loaded. Call load_courses() first."
                )

            # Generate example description and its embedding
            example_description = await self.generate_example_description(query)
            example_embedding = await self.openai_client.generate_embedding(
                example_description
            )

            # Filter courses by level and similarity
            filtered_df = (
                self.courses_df
                if levels is None
                else self.courses_df[self.courses_df["level"].isin(levels)]
            )
            filtered_df = filtered_df.reset_index(drop=True)

            similar_course_indices = self.find_similar_courses(
                filtered_df, example_embedding
            )
            filtered_df = filtered_df.iloc[similar_course_indices]

            # Prepare course string for the prompt
            course_string = "\n".join(
                f"{row['course']}: {row['description']}"
                for _, row in filtered_df.iterrows()
            )

            # Prepare the recommendation prompt
            system_rec_message = f"""You are an expert academic advisor specializing in personalized course recommendations. \
When evaluating matches between student profiles and courses, prioritize direct relevance and career trajectory fit.

Context: Student Profile ({query})
Course Options: 
{course_string}

REQUIREMENTS:
- Return exactly 10 courses, ranked by relevance and fit
- Recommend ONLY courses listed in Course Options
- If a course is cross-listed, write the course number as "COURSEXXX (Cross-listed as COURSEYYY)"
- For each recommendation include:
  1. Course number (include cross-listed courses)
  2. Course name
  2. Two-sentence explanation focused on student's specific profile/goals
  3. Confidence level (High/Medium/Low)

FORMAT (Markdown):
1. **COURSEXXX: COURSE_TITLE**
Rationale: [Two clear sentences explaining fit]
Confidence: [Level]

2. [Next course...]

CONSTRAINTS:
- NO general academic advice
- NO mentions of prerequisites unless explicitly stated in course description
- NO suggestions outside provided course list
- NO mention of being an AI or advisor"""

            messages = [{"role": "system", "content": system_rec_message}]

            response = await self.openai_client.client.chat.completions.create(
                model=self.openai_client.rec_model,
                messages=messages,
                temperature=0,
                stream=False,
            )
            recommendation = response.choices[0].message.content
            return recommendation
        except Exception as e:
            return f"Error: {str(e)}"
