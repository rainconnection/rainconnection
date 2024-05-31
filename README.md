# 각종 코드 저장용 레포.
폴더 형태 변경으로 조금 문제 있을 수 있다.

### RRF
Reciprocal rank fusion
* 다양한 ranking model output을 조합하는데에 유용한 듯.
* stat_base, model_base ranker를 모두 agg할 수 있겠다.
* 각 모형의 output을 input으로 넣는 형태로 생각했다.

## local test
sh bin/setup.sh
source venv/bin/activate

python src/thompson_simulation.py