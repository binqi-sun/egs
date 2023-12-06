import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from agents.ppo_agent import PpoAgent
from envs.dag_sched_env import EGSEnv
from utils.device import config_device
from utils.args import get_args



def egs(agent: PpoAgent, env: EGSEnv, policy_name="ppo"):
    policy = agent.policy if policy_name == "ppo" else agent.random_policy
    observation, is_end = env.reset()

    while not is_end:
        action, _ = policy(observation)
        observation, is_end = env.step(int(action))

    if env.workers is not None and env.workers < env.get_result():
        print(
            f"EGS: (Warning) input DAG cannot be scheduled with {env.workers} workers, \
            {env._dag_width} workers are used instead."
        )
    else:
        print(f"EGS: input DAG is scheduled with {env._dag.width} workers.")

    env.save_dot()


def main():
    args = get_args()
    
    config_device(args.gpu_id)

    env = EGSEnv(args.in_dot, out_path=args.out_dot, workers=args.workers)
    
    actor = None
    try:
        actor = tf.saved_model.load(args.model)
    except:
        print(f"EGS: Failed to load pre-trained model or no pre-trained model is specified. Random policy will be used instead.")
    
    agent = PpoAgent(actor)
    
    egs(agent, env)



if __name__ == "__main__":
    main()
