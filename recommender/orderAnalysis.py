import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List

def extract_course_numbers(recommendation: str) -> List[str]:
    """Extract course numbers from recommendation text"""
    pattern = r'\*\*([\w\s]+?\d{3})'
    matches = re.findall(pattern, recommendation)
    return [match.strip() for match in matches]

async def run_analysis(recommender, query: str, n_trials: int = 10, levels: List[int] = None) -> Dict:
    """
    Run multiple trials of course recommendations and analyze the similarity rankings.
    
    Args:
        recommender: Initialized recommender model
        query: Student query string
        n_trials: Number of recommendation trials to run
        levels: Optional list of course levels to filter by
    
    Returns:
        Dictionary containing rank counts and all recommendations
    """
    rank_counts = defaultdict(int)
    all_recommendations = []
    
    for i in range(n_trials):
        print(f"Running trial {i+1}/{n_trials}")
        
        # Get recommendation and sorted courses
        recommendation, sorted_df = await recommender.recommend(query, levels)
        
        # Extract recommended courses and find their ranks
        recommended_courses = extract_course_numbers(recommendation)
        ranks = []
        for course in recommended_courses:
            course_rank = sorted_df[sorted_df['course'] == course]['similarity_rank'].iloc[0]
            rank_counts[course_rank] += 1
            ranks.append(course_rank)
        
        # Store full recommendation
        all_recommendations.append({
            'recommendation': recommendation,
            'recommended_courses': recommended_courses,
            'ranks': ranks
        })
    
    # Plot the results
    ranks = sorted(rank_counts.keys())
    counts = [rank_counts[r] for r in ranks]
    
    plt.figure(figsize=(15, 6))
    plt.bar(ranks, counts)
    plt.title(f"Distribution of Recommended Course Ranks\n(Query: {query[:50]}{'...' if len(query) > 50 else ''})")
    plt.xlabel('Similarity Rank (0 = Most Similar)')
    plt.ylabel('Number of Times Recommended')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Print summary statistics
    total_recommendations = sum(counts)
    print(f"\nSummary Statistics:")
    print(f"Total recommendations: {total_recommendations}")
    print(f"Unique ranks recommended: {len(ranks)}")
    print(f"Most frequently recommended rank: {max(rank_counts.items(), key=lambda x: x[1])[0]}")
    print(f"Average rank: {sum(r * c for r, c in rank_counts.items()) / total_recommendations:.2f}")
    
    return {
        'rank_counts': dict(rank_counts),
        'all_recommendations': all_recommendations
    }