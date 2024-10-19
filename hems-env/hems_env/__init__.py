from gymnasium.envs.registration import register

register(id="hems_env/HouseholdEnv-v0", entry_point="hems_env.envs:HouseholdEnv")
