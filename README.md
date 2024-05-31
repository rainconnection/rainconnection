# 추천

## 실시간 추천
![image](https://github.com/rainconnection/recommend/assets/130852983/dee765e3-5410-4fa4-9349-82a224dc45b2)
* 출처 : Deep Neural Networks for YouTube Recommendations(2016), Paul Covington 등
* user cluster를 미리 나눠놓고(배치) 해당 cluster를 대상으로 추천(실시간)하는 방식이면 속도를 잡을 수 있을 듯
* 결국 속도가 중요한데..
* user topic이라고 부른다. topic modeling 기반으로 user clustering이 가능한가보다.
* user, feature embeddin

### RRF
Reciprocal rank fusion
* 다양한 ranking model output을 조합하는데에 유용한 듯.
* stat_base, model_base ranker를 모두 agg할 수 있겠다.
* 각 모형의 output을 input으로 넣는 형태로 생각했다.

## local test
sh bin/setup.sh
source venv/bin/activate

python src/thompson_simulation.py