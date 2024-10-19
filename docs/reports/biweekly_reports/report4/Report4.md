# Training Prosumer Agents with Reinforcement Learning.

***

> >> **Biweekly Report 4.** ( $25^{th} Apr - 6^{th} May : 2024$ )

***

![](/home/tux_term/.var/app/com.github.marktext.marktext/config/marktext/images/2024-05-06-18-53-46-image.png)

***

## 1. Status/ Progress

### Current Iteration

- [x] simplified reward functions

- [x] policy gradient learn/inference with SAC

- [x] organized episodic experiences as csv

- [x] improvement in structure, tests, & typos  fixes

### Next Iteration (Plan)

- [ ] vectorized environment for parallel learning

- [ ] test out integration of replay buffer for experience replay

---

## 2. Components

Taking feedback from weekly catchups into account and updated understanding of the system following changes were made to different components.

### 2.1. Environment Development

2.1.1. **Rewards**

- reward functions were simplified as suggested and run

  - **discouraging the difference as** $-abs(\text{net exchange})$

  - discouraging the difference as  $-(\text{net exchange})^{2}$  given,

  - $\text{net exchange} = \text{Power Household} + \text{Power PV} + \text{action}$

- the resulting grahics for **minimized absolute exchange** as reward is in the Graphics/Logs section.

2.1.2 **Experiences**

- CSV files were extracted per episode with original timestep indices during sampling action.

### 2.2. Battery Module

- Minor fixes were made to the issue where the return of  `get_current_soc` method was not as intended.

### 2.3. Agent

- Algorithm in use:
  - Proximal Policy Optimization
  - Soft Actor Critic

### 2.4. Miscellaneous

- Improvement in running main.py by adding argument parser: `python main.py --learn --save_stats --save_graphics` to run the policy learning, saving agents experiences to action that were sampled from the policy, and saving  resulting figures from experiences/dataset respectively as shown in the section Graphics/Logs below.

- Added test for environment and battery units.

### 2.5. Graphics/ Logs

```python
hyper_params: dict = {
    "dataset_len": total_dataset
    "learning_rate": 0.0003,
    "total_timesteps": 60000
    "episode_limit": 50
}
```

#### During Learning

##### PPO

**Value Loss** : Error between value function outputs and estimate

![](./imgs/ppo/learn_tb_value_loss.png)

**Policy Gradient Loss** :  Loss function policy network uses

![](./imgs/ppo/learn_tb_policy_gradient_loss.png)

**Approximate KL** : Approximate mean KL divergence b/w policies

![](./imgs/ppo/learn_tb_approx_kl.png)

**Clip Range** : clip range of surrogate loss of PPO

![](./imgs/ppo/learn_tb_clip_range.png)

**Clip Fraction** : average of clipped / inside the clipping region

![](./imgs/ppo/learn_tb_clip_fraction.png)

**Entropy Loss** : Mean value of Entropy loss

![](./imgs/ppo/learn_tb_entropy_loss.png)

**Explained Variance**: Fraction of the return variance explained by the value function

![](./imgs/ppo/learn_tb_explained_variance.png)



##### SAC

> The tensorboard logging for SAC model is still to be resolved

____

#### During Sampling

**Generated PV/ Demanded Household**

- Before normalization/ After Denormalization
  ![](./imgs/pw_household_pv_original_line.png)

- After normalization / Before Denormalization
  ![](./imgs/pw_household_pv_norm_line.png)

**PPO** 

Tensorboard Logs with Smoothing( Exponential Mooving Average )

- Inputs
  ![](./imgs/ppo/result_tb_power_pv.png)
  ![](./imgs/ppo/result_tb_power_household.png)
  ![](./imgs/ppo/result_tb_soc.png)

- Action
  ![](./imgs/ppo/result_tb_action.png)

- Reward
  ![](./imgs/ppo/result_tb_rewards.png)

- Exchanges
  ![](./imgs/ppo/result_tb_exchanges.png)

- Overall experiences
  ![](./imgs/ppo/overall_experience.png)

- Observation relative action
  ![](./imgs/ppo/observation_relative_action.png)

- Observation relative rewards
  ![](./imgs/ppo/observation_relative_rewards.png)

- Cumulative rewards
  ![](./imgs/ppo/cumulative_rewards.png)

**SAC**

- Tensorboard Logs with Smoothing( Exponential Mooving Average )

  - Inputs
    ![](./imgs/sac/result_tb_power_household.png)
    ![](./imgs/sac/result_tb_power_pv.png)
    ![](./imgs/sac/result_tb_soc.png)

  - Action
    ![](./imgs/sac/result_tb_action.png)

  - Reward`
    ![](./imgs/sac/result_tb_rewards.png)

  - Exchanges
    ![](./imgs/sac/result_tb_exchanges.png)   

- Overall experiences
  ![](./imgs/sac/overall_experience.png)

- Observation relative action
  ![](./imgs/sac/observation_relative_action.png)

- Observation relative rewards
  ![](./imgs/sac/observation_relative_rewards.png)

- Cumulative rewards
  ![](./imgs/sac/cumulative_rewards.png)

---

## 5. References

- [1. ] [SAC &mdash; Stable Baselines3 2.3.2 documentation](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
- [2. ] Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor [arXiv:1801.01290](https://arxiv.org/abs/1801.01290) [cs.LG]
