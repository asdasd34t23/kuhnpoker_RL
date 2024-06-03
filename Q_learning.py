import random
import numpy as np
import matplotlib.pyplot as plt

#쿤 포커
class KuhnPoker:
    def __init__(self):
        self.cards = [1, 2, 3]
        #편의성을 위해 j=1 Q=2 K=3 으로 설정

    def deal(self):
        random.shuffle(self.cards) #랜덤으로 카드 섞기
        return self.cards[0], self.cards[1] #플레이어1[0] ,플레이어2[1]에게 카드 배분


    def play_round(self, player1_action, player2_action, player1_card, player2_card):
        if player1_action == 'bet': #플1이 베팅하고
            if player2_action == 'call': #플2가 콜한다면
                return 1 if player1_card > player2_card else -1 #플1이 높으면 -1 리턴
            else:  # 플2 폴드 -> 플1승
                return 1 #1 리턴
        else:  # 플1 체크
            if player2_action == 'bet': #플2가 베팅
                return 1 if player2_card > player1_card else -1 #플레이어2가 이기면 +1 지면 -1 리턴
            else:  # 플2 체크
                return 1 if player1_card > player2_card else -1 #플레이어2가 이기면 +1 지면 -1 리턴


#Q-러닝 알고리즘
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.01):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}
    '''
    alpha: 학습률(learning rate). Q 값을 업데이트할 때 현재 Q 값과 새로운 정보의 비율을 조절한다. 
        값이 0에 가까울수록 기존 정보를 더 신뢰하고, 1에 가까울수록 새로운 정보를 더 신뢰한다.
        
    gamma: 할인율(discount factor). 미래 보상에 대한 현재 가치의 중요성을 조절한다. 
        값이 0에 가까울수록 현재 보상을 더 중시하고, 1에 가까울수록 미래 보상을 더 중시한다.
        
    epsilon: 탐색률(exploration rate). 무작위로 행동을 선택할 확률. 초기에는 1.0으로 설정되어 탐색을 최대화한다.

    epsilon_decay: 탐색률 감소 비율. 에피소드가 진행될수록 탐색률을 줄여서 점차 최적의 행동을 선택하도록 합니다.

    epsilon_min: 최소 탐색률. 탐색률이 이 값 이하로는 감소하지 않게 설정한다.

    q_table: 상태-행동 쌍의 Q 값을 저장하는 딕셔너리. 예를 들어, 상태와 행동의 조합이 키(key)이고, Q 값이 값(value)이다.
    '''
    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    '''
    get_q_value 메서드는 주어진 상태와 행동에 대한 Q 값을 반환한다.
    state: 현재 상태.
    action: 현재 상태에서의 행동.
    q_table에 해당 상태-행동 쌍의 Q 값이 없다면 기본값으로 0.0을 반환한다.
    '''

    def update_q_value(self, state, action, reward, next_state): #update_q_value 메서드는 주어진 상태-행동 쌍의 Q 값을 업데이트
        '''
         state: 현재 상태.
        action: 현재 상태에서의 행동.
        reward: 현재 상태-행동 쌍에서 얻은 보상
        next_state: 다음 상태.
        '''
        best_next_action = self.get_best_action(next_state) #best_next_action: 다음 상태에서의 최적 행동.

        td_target = reward + self.gamma * self.get_q_value(next_state, best_next_action)
        #td_target: 목표 값(타겟 값)으로, 보상과 다음 상태의 최적 Q 값을 합한 값.

        td_error = td_target - self.get_q_value(state, action)
        #td_error: TD 오차(Temporal Difference Error)로, 목표 값과 현재 Q 값의 차이.

        new_q_value = self.get_q_value(state, action) + self.alpha * td_error
        #new_q_value: 업데이트된 Q 값으로, 기존 Q 값에 학습률을 곱한 TD 오차를 더한 값.

        self.q_table[(state, action)] = new_q_value
        #업데이트된 Q 값을 q_table에 저장함.



    def get_best_action(self, state): #주어진 상태에서 가장 높은 Q 값을 가지는 최적 행동을 반환한다. state는 현재상태.
        actions = ['bet', 'check'] #actions: 가능한 행동 목록.

        q_values = [self.get_q_value(state, action) for action in actions] #q_values: 각 행동에 대한 Q 값 목록.

        max_q_value = max(q_values) #max_q_value: 가장 높은 Q 값.

        best_actions = [action for action, q_value in zip(actions, q_values) if q_value == max_q_value]
        #best_actions: 가장 높은 Q 값을 가지는 행동들의 목록.

        return random.choice(best_actions) #여러 최적 행동이 있을 경우, 무작위로 하나를 선택함.


    def choose_action(self, state): #  choose_action 메서드는 탐험(무작위 선택)과 활용(Q 값에 따른 최적 행동) 사이에서 결정함. state: 현재 상태.
        if np.random.uniform(0, 1) < self.epsilon:

            return random.choice(['bet', 'check'])
        else:
            return self.get_best_action(state)
        # np.random.uniform(0, 1) < self.epsilon 조건이 참이면 무작위 행동을 선택하고, 그렇지 않으면 최적 행동을 선택함.


    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            #decay_epsilon 메서드는 epsilon 값을 점차 감소시켜 탐험의 비율을 줄인다.
            #epsilon 값이 epsilon_min보다 크면, epsilon에 epsilon_decay를 곱하여 감소시킨다.


#플레이어 1 학습시키기
def train(agent, game, episodes=40000, window=4000):
    win_rates = [] # 승률을 저장할 리스트
    total_wins = 0 # 총 승리 횟수 초기화
    win_buffer = [] # 최근 window 크기만큼의 승리 여부를 저장할 버퍼

    for episode in range(1, episodes + 1): # 각 에피소드에 대해 반복
        player1_card, player2_card = game.deal() # 플레이어1과 플레이어2에게 카드 배분
        state = (player1_card, 'start') # 초기 상태 설정

        player1_action = agent.choose_action(state) # 플레이어1이 행동 선택
        if player1_action == 'bet': # 만약 플레이어1이 베팅을 선택하면
            player2_action = 'call' if random.random() < 0.5 else 'fold' # 플레이어2는 50% 확률로 콜하거나 폴드
        else:
            player2_action = 'check' # 플레이어1이 체크하면, 플레이어2도 체크

        reward = game.play_round(player1_action, player2_action, player1_card, player2_card) # 게임 라운드를 진행하고 보상을 얻음
        next_state = (player1_card, player1_action) # 다음 상태 설정
        agent.update_q_value(state, player1_action, reward, next_state) # Q 값을 업데이트

        if reward > 0: # 만약 보상이 양수이면 (승리한 경우)
            win_buffer.append(1) # 버퍼에 1 추가
        else: # 보상이 음수이면 (패배한 경우)
            win_buffer.append(0) # 버퍼에 0 추가

        if len(win_buffer) > window: # 버퍼의 크기가 window를 초과하면
            win_buffer.pop(0) # 가장 오래된 값을 제거

        agent.decay_epsilon() # epsilon 값을 감소시켜 탐험 비율을 줄임

        if episode % window == 0: # 에피소드가 window의 배수일 때
            win_rate = np.mean(win_buffer) #현재 window의 평균 승률을 계산
            win_rates.append(win_rate) #승률 리스트에 추가

    return win_rates #승률 리스트를 반환


'''
플레이어1 학습시키기:
플레이어1은 Q-러닝 알고리즘을 사용하여 자신의 전략을 학습한다.
주어진 상태에서 최적의 행동을 선택하거나 무작위로 행동을 선택하는 방식으로 학습한다.

플레이어 2는자신의 패와 무관하게 랜덤으로 행동함
플레이어 1이 베팅하면, 플레이어 2는 50% 확률로 콜하거나 폴드
플레이어 1이 체크하면, 플레이어 2도 체크
'''

# 실행 및 결과 출력
game = KuhnPoker()
agent = QLearningAgent()
win_rates = train(agent, game, episodes=40000, window=4000)

'''
game과 agent를 초기화하고,train함수를 호출해 학습을 진행시킴
시행횟수는 4만번
'''

'''
q러닝 알고리즘은 너무 난이도가 높아서 gpt를 사용함
'''

plt.plot(np.arange(1, len(win_rates) + 1) *4000, win_rates)

plt.xlabel('Episodes')
plt.ylabel('Win Rate')
plt.title('Q-Learning Performance over Time')
plt.grid(True)
plt.show()
#학습 결과를 시각화하여 에피소드에 따른 승률 변화를 그래프로 출력
