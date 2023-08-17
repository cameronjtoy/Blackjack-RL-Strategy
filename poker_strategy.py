import gym
import random
import numpy as np

class CustomBlackjackEnv(gym.Env):
    def __init__(self, decks=1, buy_in=25, initial_balance=500):
        self.decks = decks
        self.deck = self.shuffle_deck()
        self.buy_in = buy_in
        self.agent_balance = initial_balance
        self.action_space = gym.spaces.Discrete(2)  # Hit or Stand
        self.observation_space = gym.spaces.Tuple((gym.spaces.Discrete(32), gym.spaces.Discrete(12), gym.spaces.Discrete(2)))  # Player sum, dealer card, usable ace


    def shuffle_deck(self):
        deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * 4 * self.decks
        random.shuffle(deck)
        return deck

    def deal_card(self):
        return self.deck.pop()

    def step(self, action):
        reward = 0  # Initialize reward as 0, can be overridden later
        if action == 0:  # Hit
            self.player_hand.append(self.deal_card())
            if sum(self.player_hand) > 21 and 11 in self.player_hand:
                self.player_hand.remove(11)
                self.player_hand.append(1)
            if sum(self.player_hand) > 21:
                self.done = True
                reward = -1
        elif action == 1:  # Stand
            while sum(self.dealer_hand) <= 16:
                self.dealer_hand.append(self.deal_card())
                if sum(self.dealer_hand) > 21 and 11 in self.dealer_hand:
                    self.dealer_hand.remove(11)
                    self.dealer_hand.append(1)
            self.done = True
            if sum(self.player_hand) > sum(self.dealer_hand):
                reward = 1
            elif sum(self.player_hand) == sum(self.dealer_hand):
                reward = 0
            else:
                reward = -1
        if self.done:
            if reward == 1:
                reward = self.buy_in
                self.agent_balance += self.buy_in
            elif reward == -1:
                reward = -self.buy_in
                self.agent_balance -= self.buy_in
            else:
                reward = 0
        usable_ace = int(11 in self.player_hand)
        state = (sum(self.player_hand), self.dealer_hand[0] - 2, usable_ace)
        return state, reward, self.done, {'agent_balance': self.agent_balance}

    def reset(self):
        self.player_hand = [self.deal_card()]
        self.dealer_hand = [self.deal_card()]
        self.player_hand.append(self.deal_card())
        self.dealer_hand.append(self.deal_card())
        self.done = False

        if len(self.deck) <= (52 * self.decks) / 2:
            self.deck = self.shuffle_deck()

        if self.agent_balance < self.buy_in:
            print("Agent doesn't have enough balance for buy-in!")
            return None

        self.agent_balance -= self.buy_in
        usable_ace = int(11 in self.player_hand)
        state = (sum(self.player_hand), self.dealer_hand[0] - 2, usable_ace)
        self.done = False
        return state

    def render(self, mode='human'):
        print(f"Player hand: {self.player_hand}, Dealer hand: {self.dealer_hand}, Agent balance: {self.agent_balance}")



def q_learning(env, total_episodes, alpha=0.85, gamma=0.95, epsilon=1, epsilon_decay=0.9777):
    Q = np.zeros([env.observation_space[0].n, env.observation_space[1].n, env.observation_space[2].n, env.action_space.n])

    for episode in range(total_episodes):
        state = env.reset()
        if state is None:
            print("Skipping episode due to insufficient balance")
            continue
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state[0], state[1], state[2]])

            next_state, reward, done, _ = env.step(action)

            Q[state[0], state[1], state[2], action] = Q[state[0], state[1], state[2], action] + \
                alpha * (reward + gamma * np.max(Q[next_state[0], next_state[1], next_state[2]]) - Q[state[0], state[1], state[2], action])

            state = next_state
        
        # Apply epsilon decay at the end of the episode
        epsilon *= epsilon_decay

    return Q



def sarsa(env, episodes, alpha=0.1, gamma=0.9, epsilon=0.1):
    Q = np.zeros([env.observation_space[0].n, env.observation_space[1].n, env.observation_space[2].n, env.action_space.n])

    for episode in range(episodes):
        state = env.reset()
        if state is None:
            print("Skipping episode due to insufficient balance")
            continue
        done = False

        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state[0], state[1], state[2]])

        while not done:
            next_state, reward, done, _ = env.step(action)

            if random.uniform(0, 1) < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state[0], next_state[1], next_state[2]])

            Q[state[0], state[1], state[2], action] = Q[state[0], state[1], state[2], action] + \
                alpha * (reward + gamma * Q[next_state[0], next_state[1], next_state[2], next_action] - Q[state[0], state[1], state[2], action])

            state = next_state
            action = next_action

    return Q

def main():
    decks = 1
    episodes = 10
    env = CustomBlackjackEnv(decks)

    # Train using Q-learning
    Q_q_learning = q_learning(env, episodes)
    print("Q-learning Q-table:", Q_q_learning)

    # # Train using SARSA
    # Q_sarsa = sarsa(env, episodes)
    # print("SARSA Q-table:", Q_sarsa)

if __name__ == "__main__":
    main()
