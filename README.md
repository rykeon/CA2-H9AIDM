# CA2-H9AIDM
NCI MSc in Artificial Intelligence
Reinforcement Learning surge pricing strategy.

## Project Structure
```
C:.
|   .gitignore
|   README.md
|   
+---data (Raw datasets not tracked by Git)
|       cab_rides.csv
|       weather.csv
|       
+---notebooks
|       RL_SurgePricing.ipynb
+---presentation
\---report
```

## Datasets
https://www.kaggle.com/datasets/ravi72munde/uber-lyft-cab-prices/data

**Note:** Data files are not tracked in this repository due to file size
Download the dataset from the link above and place the files in the `data/` folder.

## Project Progress

### Phase 1 — Data Exploration and Cleaning — Done (Cells 1-10.5)
- Loaded cab_rides (693,071 rides) and weather (6,276 records)
- Removed 55,095 rides with null prices
- Converted timestamps to datetime
- Explored unique zones, ride types, price and surge distributions
- Merged both datasets by hour and zone
- Fixed duplicate merge with hourly weather aggregation
- Final clean dataset: 636,992 rides, 21 columns, zero nulls

### Phase 2 — Demand Model — In Progress
- Define demand (count rides per zone, hour, and conditions)
- Engineer features for regression
- Train regression model (Random Forest)
- Evaluate model performance
- Save model for use in RL environment

### Phase 3 — RL Environment and Agents — Pending
- Define MDP formally (state, action, reward, transitions)
- Build custom RL environment using demand model
- Implement Algorithm 1: Q-Learning
- Implement Algorithm 2: SARSA
- Compare against baselines (no surge, fixed surge)

### Phase 4 — Report and Presentation — Pending
- Write 8-page PDF report (IEEE format)
- Record 10-minute video presentation

## Algorithms
- Algorithm 1: Q-Learning
- Algorithm 2: SARSA
