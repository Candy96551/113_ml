import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)

episode_durations = []  # 儲存每次撐多久的紀錄
duration = 0

for _ in range(1000):  # 設大一點可以玩很多局
    env.render()
    cart_pos, cart_vel, pole_angle, pole_vel = observation

    # 固定策略：根據竿子角度與角速度決定方向
    if pole_angle > 0:
        action = 1  # 推右
    else:
        action = 0  # 推左

    observation, reward, terminated, truncated, info = env.step(action)
    duration += 1

    if terminated or truncated:
        print(f"撐了 {duration} 步")
        episode_durations.append(duration)
        duration = 0
        observation, info = env.reset()

env.close()

print("每一局撐的時間：", episode_durations)
print("平均撐的時間：", sum(episode_durations) / len(episode_durations))
