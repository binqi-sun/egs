import numpy as np
import tensorflow as tf


class PpoAgent:
    def __init__(self, actor=None):
        self.actor = actor
    
    @tf.function
    def policy(self, observation):
        action_mask = observation["act_msk"]
        if self.actor is None:
            advantages = tf.random.uniform([1, tf.shape(action_mask)[-1]])
        else:
            advantages = self.actor(observation["nn_input"])
        masked_advantages = tf.where(action_mask, advantages, -np.inf)
        act_probs = tf.nn.softmax(masked_advantages)
        action = tf.argmax(act_probs, axis=-1)
        return action, act_probs  # (1,)

