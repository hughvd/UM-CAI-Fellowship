import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

class graphTool(object):
    """A tool to load a dataframe of vectors and generate visualizations. 
    """

    def __init__(self, df: pd.DataFrame):
        """Loads dataframe.
        """
        super().__init__()
        print('Initializing...')
        self.df = df

        # Make similarity matrix
        matrix = [vec for vec in self.df['vector']]
        # Compute cosine similarities
        self.similarity_matrix = cosine_similarity(matrix)
        #Find similar courses
        self.deps = self.df['department'].tolist()

       