from tqdm import tqdm
# from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.tensorboard import SummaryWriter

from evaluation.evaluate import evaluate_policy

from utils.event_logger import logger
from utils.configs import policy_config


def retrain_policy(
    policy, train_env, eval_env, model_used, tensorboard_log_dir, policy_checkpoint_dir
):
    """
    Retrains the policy

    Args:
        policy: policy object
        train_env: training environment
        eval_env: evaluation environment
        model_used: model used
        tensorboard_log_dir: tensorboard log directory
        policy_checkpoint_dir: policy checkpoint directory
    Returns:
        A list of accumulated mean rewards
    """
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    logger.info(
        f">> retraining/evaluating {policy_config['num_retrain_eval_cycles']} cycles ..."
    )
    policy = policy.load(policy_checkpoint_dir)
    logger.info(">> policy loaded")

    rewards_accumulator = []
    cost_accumulator = []
    for i in tqdm(range(policy_config["num_retrain_eval_cycles"])):
        train_env.reset()
        eval_env.reset()
        policy.set_env(train_env)
        logger.info(f"> retraining ...")
        policy.learn(
            total_timesteps=policy_config["retrain_timesteps"],
            tb_log_name=f"{model_used}_retrain_run",
            reset_num_timesteps=policy_config["reset_num_timesteps"],
        )
        logger.info(f"> evaluating ...")
        mean_reward, mean_cost = evaluate_policy(
            policy, 
            eval_env, 
            n_eval_episodes=policy_config["num_eval_episodes"],
            deterministic=True,
        )
        rewards_accumulator.append(mean_reward)
        cost_accumulator.append(mean_cost)
        writer.add_scalar("mean_reward", mean_reward, i)
        writer.add_scalar("mean_cost", mean_cost, i)
        logger.info(f"mean reward: {mean_reward}, mean cost: {mean_cost} iteration: {i}")
        logger.info("> saving policy checkpoint...")
        policy.save(policy_checkpoint_dir)

    logger.info(">> evaluation graphics saved")
    writer.close()
    train_env.close()
    eval_env.close()
    logger.info(">> environment closed")

    return (
        rewards_accumulator
        , cost_accumulator
    )
