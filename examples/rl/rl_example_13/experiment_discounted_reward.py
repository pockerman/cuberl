GAMMA = 0.99

def approach_1(rewards: list) -> list:
    R = 0
    discounted_rewards = []
    
    # Discount future rewards back to the present using gamma
    for r in rewards[::-1]:
        R = r + GAMMA * R
        discounted_rewards.insert(0,R)

    return discounted_rewards

def approach_2(rewards: list) -> list:

    discounted_rewards = []
    for t in range(len(rewards)):
        G = 0.0
        for k, r in enumerate(rewards[t:]):
            G += (GAMMA**k)*r
        discounted_rewards.append(G)
    return discounted_rewards

def approach_3(rewards: list) -> list:

    res = []
    discounted_rewards = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
        
    return list(reversed(res)) 


if __name__ == '__main__':


    rewards = [float(i) for i in range(5)]

    print(f'Rewards: {rewards}')
    print(f'rewards[::-1], {rewards[::-1]}')

    one = approach_1(rewards)

    print(f'one {one}')

    two = approach_2(rewards)
    print(f'two {two}')

    three = approach_3(rewards)
    print(f'three {three}')

