from gymnasium.envs.registration import register

register(
    id="HighwayDriving-v0",
    entry_point="env.highway_env:CarRacingDrivingEnv",
)
