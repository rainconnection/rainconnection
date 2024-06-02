
from typing import List

class rrf_reranker:
    
    def __init__(self, k = 10):
        self.k = k

    def fusion(self, items : List[str], ranks : List[List[int]]):
        
        rank_score_dict = {}
        for item in items:
            score = 0.0
            for rank in ranks:
                score += 1.0 / (k + rank)

            rank_score_dict[item] = score

        return sorted(rank_score_dict.items(), key = lambda x : x[1], reverse = True)