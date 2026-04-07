# CA2-H9AIDM
NCI MSc in Artificial Intelligence
Reinforcement Learning surge pricing strategy.

## Project Structure
```
C:.
|   .gitignore
|   README.md
|   
+---data
|       cab_rides.csv
|       weather.csv
|       
+---notebooks
|   |   RL_SurgePricing_version.ipynb
|   |   surge_pricing_utils.py
|   |   
|   +---outputs
|   |       action_distribution_fitted_q-iteration.png
|   |       action_distribution_sarsa.png
|   |       action_distribution_tabular_q-learning.png
|   |       average_action_by_policy.png
|   |       best_policy_action_heatmap.png
|   |       context_table.csv
|   |       cumulative_reward_over_test_horizon.png
|   |       fqi_training_curve.png
|   |       historical_price_surge_by_hour.png
|   |       historical_surge_by_hour.png
|   |       price_distribution.png
|   |       q_learning_training_curve.png
|   |       reward_vs_action.png
|   |       sarsa_training_curve.png
|   |       supervised_model_diagnostics.png
|   |       supervised_model_metrics.csv
|   |       surge_multiplier_distribution.png
|   |       test_policy_comparison_vs_no_surge.csv
|   |       test_policy_step_results.csv
|   |       test_policy_summary.csv
|   |       test_total_reward_by_policy.png
|   |       thompson_sampling_training_curve.png
|   |       top_routes_by_best_policy_action.png
|   |       top_routes_by_historical_surge.png
|   |       training_curve_comparison.png
|   |       validation_policy_summary.csv
|   |       
|   \---__pycache__
|           surge_pricing_utils.cpython-313.pyc
|           
+---presentation
\---report

```

## Datasets
https://www.kaggle.com/datasets/ravi72munde/uber-lyft-cab-prices/data


## Algorithms
- Algorithm 1: Tabular Q-Learning
- Algorithm 2: Fitted Q-Iteration (FQI)
