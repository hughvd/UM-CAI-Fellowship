import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List


def extract_course_numbers(recommendation: str) -> List[str]:
    """Extract course numbers from recommendation text"""
    pattern = r"\*\*([\w\s]+?\d{3})"
    matches = re.findall(pattern, recommendation)
    return [match.strip() for match in matches]


def plot_rank_and_course(rank_counts, course_rank_counts, query):
    """Plot rank distribution and course rank distribution"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    ranks = sorted(rank_counts.keys())
    counts = [rank_counts[r] for r in ranks]

    # Plot 1: Rank Distribution
    ax1.bar(ranks, counts)
    ax1.set_title(
        f"Distribution of Recommended Course Ranks\n(Query: {query[:50]}{'...' if len(query) > 50 else ''})"
    )
    ax1.set_xlabel("Similarity Rank (0 = Most Similar)")
    ax1.set_ylabel("Number of Times Recommended")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Course-Rank Scatter
    unique_courses = sorted(course_rank_counts.keys())
    course_positions = range(len(unique_courses))

    # Create scatter plot points
    scatter_points = []
    for course_idx, course in enumerate(unique_courses):
        for rank, count in course_rank_counts[course].items():
            scatter_points.append((course_idx, rank, count))

    if scatter_points:
        x, y, sizes = zip(*scatter_points)
        max_size = max(sizes)
        # Scale sizes for visibility (adjust multiplier as needed)
        scaled_sizes = [100 * (s / max_size) ** 0.5 for s in sizes]

        scatter = ax2.scatter(x, y, s=scaled_sizes, alpha=0.6)

        # Add count labels to dots
        for i, (x_pos, y_pos, count) in enumerate(scatter_points):
            ax2.annotate(
                str(count),
                (x_pos, y_pos),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
    ax2.set_title("Course Recommendations by Similarity Rank")
    ax2.set_xlabel("Recommended Course")
    ax2.set_ylabel("Similarity Rank")
    ax2.set_xticks(course_positions)
    ax2.set_xticklabels(unique_courses, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3)

    return fig


def plot_rank(ranks, counts, figsize, title):
    # Plot 1: Rank Distribution
    plt.figure(figsize=figsize)
    plt.bar(ranks, counts)
    plt.title(title)
    plt.xlabel("Similarity Rank (0 = Most Similar)")
    plt.ylabel("Number of Times Recommended")
    plt.grid(True, alpha=0.3)

    return plt.gcf()


async def run_analysis(
    recommender, query: str, n_trials: int = 10, levels: List[int] = None
) -> Dict:
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
    # {course: {rank: count}}
    course_rank_counts = defaultdict(lambda: defaultdict(int))

    for i in range(n_trials):
        print(f"Running trial {i+1}/{n_trials}")

        # Get recommendation and sorted courses
        recommendation, sorted_df = await recommender.recommend(query, levels)

        # Extract recommended courses and find their ranks
        recommended_courses = extract_course_numbers(recommendation)
        ranks = []
        for course in recommended_courses:
            course_rank = sorted_df[sorted_df["course"] == course][
                "similarity_rank"
            ].iloc[0]
            rank_counts[course_rank] += 1
            course_rank_counts[course][course_rank] += 1
            ranks.append(course_rank)

    fig = plot_rank_and_course(
        rank_counts=rank_counts, course_rank_counts=course_rank_counts, query=query
    )

    return {
        "rank_counts": rank_counts,
        "course_rank_counts": course_rank_counts,
        "figure": fig,
    }
