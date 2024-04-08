import matplotlib.pyplot as plt
import json
import os

with open(os.getcwd()+'/mountain-car-master/Results/DDPG.json', 'r') as file:
    DDPG = json.load(file)

with open(os.getcwd()+'/mountain-car-master/Results/TD3.json', 'r') as file:
    TD3 = json.load(file)

# Extracting the score list
score_list_DDPG = DDPG["score_list"]
rolling_score_list_DDPG = DDPG["rolling_score_list"]
time_list_DDPG = DDPG["time_list"]
score_list_TD3 = TD3["score_list"]
rolling_score_list_TD3 = TD3["rolling_score_list"]
time_list_TD3 = TD3["time_list"]
epochs = range(1, len(score_list_DDPG) + 1)


plt.figure(figsize=(10, 6))
plt.plot(epochs, time_list_DDPG, linestyle='-', color='b', label='DDPG')
plt.plot(epochs, time_list_TD3, linestyle='-', color='r', label='TD3')
plt.plot(time_list_DDPG, linestyle='-', color='b')
plt.title('Comparison of DDPG and TD3 Completion Times over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Time')
plt.ylim(0, 120)
plt.grid(True)
plt.legend()

plt.figure(figsize=(10, 6))
plt.plot(epochs, score_list_DDPG, linestyle='-', color='b', label='DDPG')
plt.plot(epochs, score_list_TD3, linestyle='-', color='r', label='TD3')
plt.title('Comparison of DDPG and TD3 Scores over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.ylim(0, 90)
plt.grid(True)
plt.legend()


# plt.figure(figsize=(10, 6))
# plt.plot(epochs, score_list_DDPG, linestyle='-', color='b', label='DDPG')
# plt.plot(epochs, score_list_TD3, linestyle='-', color='r', label='TD3')
# plt.title('Comparison of DDPG and TD3 Rolling Scores over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Rolling Score (100 epoch window)')
# plt.ylim(0, 90)
# plt.grid(True)
# plt.legend()

plt.show()
