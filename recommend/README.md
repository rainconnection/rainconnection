## mab-thompson
* 기본적인 user action 기반 sampling - recommend(rank 등)
* 간단하면서도 꽤나 유용
* log 수집 환경 내에서 update는 필요없긴 하다.

## RRF
Reciprocal rank fusion
* 다양한 ranking model output을 조합하는데에 유용한 듯.
* stat_base, model_base ranker를 모두 agg할 수 있겠다.
* 각 모형의 output을 input으로 넣는 형태로 생각했다.

## local test

#### thompson simulation
python recommend/thompson_simulation.py