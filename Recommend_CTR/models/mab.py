import numpy as np
from collections import deque
class thompson_sampling:
    def __init__(self, eps, item_ids, params = None):
        ## initialize
        # indexing
        self.item_ids = item_ids # item index
        self.item_to_param = {} # item_ids to param index 업데이트를 위해 일단 만들자. 더 효율적일 수 없나?
        for i,v in enumerate(self.item_ids):
            self.item_to_param[v] = i

        # params
        self.eps = eps # random 노출할 확률
        self.n_items = len(item_ids) # #item
        # beta_dist parameter initialize
        if params: # 사전 확률 분포 지정 가능
            self.alpha = np.array([x[0] for x in params])
            self.beta = np.array([x[1] for x in params])
        else:
            self.alpha = np.ones(self.n_items)
            self.beta = np.ones(self.n_items)

    def get_dist_beta(self): # 추후에 바뀔 수 있을 것 같으니 뺴놓자.
        self.scores = np.random.beta(self.alpha, self.beta)
        self.sorted_items = np.argsort(self.scores)[::-1] # 내림차순
        self.ids_top_k = self.sorted_items[:self.n_rank]

        return self.item_ids[self.ids_top_k]

    def item_choice(self, n_display = None):
        if n_display:
            self.n_display = n_display
        else: # 들어오지 않으면 직전 n_display 사용
            if self.n_display is not None:
                print('please input n_display')
                return None
        # output trey
        self.trey = np.zeros(self.n_display) - 1 # item_key에 음수가 없다고 가정
        # random output 자리 선택
        self.rand_which = np.random.random(self.n_display) < self.eps
        self.n_rand = np.sum(self.rand_which)
        self.n_rank = (self.n_display - self.n_rand) # 상위 rank에서 꺼내온 item 수

        # Thompson Sampling from betas
        self.trey[self.rand_which == False] = self.get_dist_beta()
        
        # random 노출
        self.trey[self.rand_which] = np.random.choice(self.item_ids,self.n_rand,replace = False)

        # 중복 제거 및 다음 순위 할당, 시간복잡도 n_display^2
        
        self.trey = deque(self.trey)
        self.display_items = []
        while(self.trey):
            x = self.trey.popleft()
            if x not in self.display_items:
                self.display_items.append(int(x))
            else:
                self.trey.append(self.sorted_items[self.n_rank])  # 다음 순위 꺼내오기
                self.n_rank += 1 # ranked 개수 하나 늘어남
        
        return self.display_items

    def update(self, chosen_key):
        #- input : chosen : 선택된 item key list
        for k in self.display_items:
            if k in chosen_key:
                self.alpha[self.item_to_param[k]] += 1
            else:
                self.beta[self.item_to_param[k]] += 1