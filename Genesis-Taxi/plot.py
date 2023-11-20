import optuna
import joblib
import pandas as pd
import numpy as np
import config
study = optuna.create_study(directions=["maximize"],    pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=50, reduction_factor=3))
pickle_name = config.pickle_name
study = joblib.load(pickle_name)
# df = pd.DataFrame(columns=["IOU","M1",'eps','min_points',"voxel size"])
df = pd.DataFrame(columns=["SR","alpha","gamma","epsilon"])

print("Best trial until now:")
best_trials = study.trials
# best_trials = study.best_trials
for trial in best_trials:
    try:
        trial_out = []
        print(" Value: ", trial.values)
        trial_out.append(trial.values[0])
        # trial_out.append(trial.values[1])
        print(" Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
            trial_out.append(value)
        # trial_out = np.reshape(trial_out,(1,-1))
        # df = pd.concat([pd.DataFrame(trial_out)], ignore_index=True)
        print(trial_out)
        df = pd.concat([pd.DataFrame([trial_out], columns=df.columns), df], ignore_index=False)
    except Exception as e:
            pass 
df.to_excel('results/complete_test_optuna64.xlsx')  

print(df.describe())
targets=lambda t: (t.values[0])   
# fig = optuna.visualization.plot_pareto_front(study)
# fig = optuna.visualization.plot_pareto_front(study,target_names=["IOU","M1"])
# fig = optuna.visualization.plot_param_importances(study,target=targets, target_name=["IOU"])
# fig = optuna.visualization.plot_contour(study, params=["eps", "MP"],target=targets)
# fig = optuna.visualization.plot_slice(study, params=["eps", "MP"],target=targets)

# fig = optuna.visualization.plot_intermediate_values(study)

# fig = optuna.visualization.plot_rank(study,target=targets, target_name=["SR"])
# fig.show()