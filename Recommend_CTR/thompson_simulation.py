import numpy as np
from models.mab import thompson_sampling


if __name__ == "__main__":
    item_ids = np.array(range(1,101)) # item key는 자연수
    print("item ids(자연수 가정) : ", item_ids)

    TS_test = thompson_sampling(0.3, item_ids)
    print("Random 추출 확률 : 0.3, 모형 생성")

    test_output = TS_test.item_choice(12)
    print("12개 item 선택")
    print(test_output)