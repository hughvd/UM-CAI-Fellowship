# UMGPT Recommender

## Overview
The UMGPT Recommender is a Python-based course recommendation system housed in the `umgpt_recommender` folder. This system leverages the power of GPT-4 to recommend educational courses tailored to students' queries. It features two main scripts:

- `umgpt_recommender.py`: A course recommender system that filters and recommends courses based on student queries.
- `embedding_recommender.py`: (Please add a description for this script.)

## Contents
- `umgpt_recommender.py`: Initializes with a DataFrame containing course names and descriptions, uses GPT-4 to iteratively filter and recommend courses.
- `embedding_recommender.py`: Additional details needed.

## Usage

### Initializing the Recommender

Before using the `recommend()` function, initialize the recommender by loading your course DataFrame with columns "course" and "description":

```python
import umgpt_recommender

# Load your DataFrame here
courses_df = pd.read_csv('path_to_your_courses.csv')

recommender = umgpt_recommender.Recommender(courses_df)
```

### Getting Recommendations
To get course recommendations based on a student's query:
```python
query = "I want to learn about artificial intelligence, what are some courses that I could take?"
recommended_courses = recommender.recommend(query)
print(recommended_courses)
```
