import re
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
import textwrap

CONF_MAP = {"High": 2, "Medium": 1, "Low": 0}


def parse_recommendation(text: str) -> List[tuple]:
    parts = re.split(r"\n\s*\d+\.\s*\*\*", text)
    entries = []
    for part in parts[1:]:
        header, rest = part.split("**", 1)
        code = header.strip().split(":")[0]
        m = re.search(r"Confidence:\s*(High|Medium|Low)", rest)
        conf = m.group(1) if m else "Low"
        entries.append((code, conf))
    return entries


async def generate_confidence_matrix(
    prompts: List[str], recommender, levels: List[int], n_runs: int = 1
) -> pd.DataFrame:
    """
    Async: for each prompt, runs `recommender.recommend(prompt)` n_runs times,
    parses out the 10 recommendations each time, and returns a DataFrame:
      - index = prompts
      - columns = unique course codes
      - values = average confidence in [0,2]
    """
    all_data: Dict[str, Dict[str, float]] = {}

    for prompt in prompts:
        totals: Dict[str, float] = {}
        for _ in range(n_runs):
            rec_text = await recommender.recommend(prompt, levels)
            for code, conf in parse_recommendation(rec_text):
                totals[code] = totals.get(code, 0.0) + CONF_MAP[conf]
        # average and store
        all_data[prompt] = {c: s / n_runs for c, s in totals.items()}

    df = pd.DataFrame.from_dict(all_data, orient="index").fillna(0.0)
    # sort columns by descending mean confidence
    df = df.loc[:, df.mean().sort_values(ascending=False).index]
    return df


def plot_confidence_heatmap(
    df: pd.DataFrame,
    domain: str,
    max_courses: int = None,
    height_per_row: float = 0.5,
    width_per_col: float = 0.5,
):
    """
    Plots a heatmap with:
      - Fixed figure size
      - Wrapped prompt labels
      - Optional truncation to top-N courses
    """
    # 1) Optionally keep only the top-N courses by avg confidence
    if max_courses is not None and df.shape[1] > max_courses:
        top_cols = df.mean().nlargest(max_courses).index
        df = df[top_cols]

    # 2) Wrap the prompt labels to a fixed width
    wrapped_prompts = ["\n".join(textwrap.wrap(p, width=30)) for p in df.index]

    # 3) Create the figure with a more reasonable fixed size
    fig, ax = plt.subplots(
        figsize=(
            max(12, len(df.columns) * width_per_col),
            max(4, len(df.index) * height_per_row),
        ),
        constrained_layout=True,
    )
    im = ax.imshow(df.values, origin="lower", aspect="auto")

    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=8)

    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(wrapped_prompts, fontsize=8)

    fig.colorbar(im, ax=ax, label="Avg Confidence (0–2)")
    ax.set_title(f"Avg Confidence Heatmap for “{domain}”")

    plt.show()
