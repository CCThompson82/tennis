import unittest
import os
import sys
from unityagents import UnityEnvironment

WORK_DIR = os.environ['ROOT_DIR']
sys.path.append(WORK_DIR)


class Tests(unittest.TestCase):

    def setUp(self):
        pass

    def test_unity_env(self):
        UNITY_ENV_PATH = os.environ['UNITY_ENV_PATH']
        env = UnityEnvironment(file_name=UNITY_ENV_PATH)
        brain = env.brains[env.brain_names[0]]

