from gym.envs.registration import register

register(
    id='EnergyMountainCar-v0',
    entry_point='src.environments.energy_mountaincar:EnergyMountainCarEnv',
)