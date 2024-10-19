from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils.configs import policy_config, battery_config
from utils.event_logger import logger

GRID_DRAW_RATE_EUR = 0.33
GRID_FEED_RATE_EUR = 0.08

def sample_actions(
    policy,
    test_env,
    model_used,
    tensorboard_log_dir,
    policy_checkpoint_dir,
    random_seed,
):
    """
    Samples actions from the trained policy

    Args:
        policy: policy object
        test_env: testing environment
        model_used: model used
        tensorboard_log_dir: tensorboard log directory
        policy_checkpoint_dir: policy checkpoint directory
        random_seed: random seed
    Returns:
        replay buffer(list): A list of accumulated experience
        [power_household, power_pv, sell_price, buy_price, soc, action, reward, exchange, cost]
    """
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    logger.info(f">> testing {policy_config['num_test_episodes']} Episodes ...")
    policy = policy.load(policy_checkpoint_dir)
    logger.info(">> policy loaded")

    current_step = 0
    current_episode = 0
    replay_buffer = []
    for _ in tqdm(range(policy_config["num_test_episodes"])):

        observation, _ = test_env.reset()
        truncated = False
        terminated = False
        while not (truncated or terminated):
            action, _ = policy.predict(observation, deterministic=True)
            
            observation, reward, terminated, truncated, info = test_env.step(
                action
            )

            action = (action + 1) * (
                battery_config["max_charge_rate"] - battery_config["max_discharge_rate"]
            ) / 2 + battery_config[
                "max_discharge_rate"
            ]  # rescale action to original range

            # soc constraint for saved stats
            if test_env.unwrapped.battery.current_soc == 0.0:  # if battery empty
                action = np.clip(action, 0, battery_config["max_charge_rate"])  # only charge action
            elif test_env.unwrapped.battery.current_soc == 1.0:  # if battery full
                action = np.clip(action, battery_config["max_discharge_rate"], 0 )  # only discharge action

            replay_buffer.append(
                (
                    observation[0],  # power household
                    observation[1],  # power pv
                    # GRID_FEED_RATE_EUR,
                    # GRID_DRAW_RATE_EUR,
                    observation[2],  # auction price
                    observation[3],  # battery soc
                    action,
                    reward,
                    info["exchange"],
                    info["cost"],
                )
            )
            writer.add_scalar(
                "observation/power_household", observation[0], current_step
            )
            writer.add_scalar("observation/power_pv", observation[1], current_step)
            writer.add_scalar("observation/auction_price", observation[2], current_step)
            writer.add_scalar("observation/battery_soc", observation[3], current_step)

            writer.add_scalar("results/action", action, current_step)
            writer.add_scalar("results/reward", reward, current_step)
            writer.add_scalar("results/exchange", info["exchange"], current_step)
            writer.add_scalar("results/cost", info["cost"], current_step)

            logger.info(
                f"Episode: {current_episode}, Step: {current_step}, Action: {action}, SoC: {observation[2]}, Exchange: {info['exchange']}, Reward: {reward}"
            )
            current_step += 1
        current_episode += 1
    logger.info(f"Total steps: {current_step}, Total episodes: {current_episode}")
    test_env.close()
    logger.info(">> environment closed")

    return replay_buffer
