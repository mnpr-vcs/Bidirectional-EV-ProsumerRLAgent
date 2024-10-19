This repository contains Master Thesiswork ⬇️ undertaken for the partial fulfillment of M.S. in Information Engineering @Fachhochschule-Kiel under the supervision of Prof. Dr. Hauke Schramm and Mr. Vincenz Regener.

# Prosumer-HEMS-RLpy 

> **Objective :** Training Prosumer Agents with Reinforcement Learning for Energy and Cost Optimization


The increasing integration of solar photovoltaic (PV) systems and electric vehicles (EVs) into
residential grids presents exciting opportunities for self-consumption and grid stability.
However, optimizing energy flow between these components while considering dynamic pricing,
varying solar generation, and user preferences remains a significant challenge. This undertaking is an effort towards
one such approach using reinforcement learning (RL) algorithms to optimize self-
consumption of generated PV and cost reduction in a prosumer household ( a household that
consumes from the grid as well as produces and feeds the excess photovoltaic energy back to the
grid).

---

- [[src]](./src/), [[docs]](./docs/)
- setup dev env `cd ./scripts && ./setup_dev_env.sh`
- reproduce results: `cd ./scripts && ./run_experiments.sh`
- run model server: `cd ./scripts && ./run_model_server.sh`
- run single model experiment with env changes: `cd ./scripts && ./run_env_changes.sh`

