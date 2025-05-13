#XGBClassifier on LITNET-2020
#Adjust fraction value for different sampling size (currently at line 56)

import pandas as pd
import numpy as np
import joblib
import time
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import glob
import xgboost as xgb
from sklearn.utils import resample


# Control limit of Reinforcement Learning
max_time_limit = 600 #in seconds, 3600s = 1hr
max_iterations = 1000000
max_states = 1000000000

# define environment, range of n_estimators and range of max_depth
n_values_min = 2
# n_values_min = 500
n_values_max = 5000
max_depth_values_min = 1
#max_depth_values_min = 2
max_depth_values_max = 100

# define initial state
initial_state = (1000, 5)  # Initial state: n=70, max_depth=5
total_states = (n_values_max - n_values_min + 1) * (max_depth_values_max - max_depth_values_min + 1)



# define random_state # need to run 10 times with different random_state value.
define_random_state = 42

# Load datasets
file_names = glob.glob("../dataset_preprocessed4/*.csv")
start_time = time.time()
dataframes = [pd.read_csv(file) for file in file_names]
readCSV_time = time.time() - start_time
print(f"readCSV time: {readCSV_time}")

start_time = time.time()
data = pd.concat(dataframes, ignore_index=True)
concat_time = time.time() - start_time
print(f"concat_time: {concat_time}")
print(data.head())


###
# data_samples = data.sample(frac=0.0001, random_state=define_random_state)
fraction=0.1 #adjust here for different sampling size
num_classes=13
class_column = "attack_t"
grouped = data.groupby(class_column)
class_0 = grouped.get_group(0) if 0 in grouped.groups else pd.DataFrame()


full_size = int(len(data))
sampling_size = int(full_size * fraction)
samples_per_class = max(sampling_size // num_classes, 1)  # at least 1 sample per class

print(f"\n--- Sampling with fraction {fraction} ---")
print(f"Expected total dataset size: {sampling_size}")
print(f"Expected samples per class: {samples_per_class}")

balanced_data = []

for class_label, group in grouped:
    if len(group) >= samples_per_class:
        sampled_group = resample(group, n_samples=samples_per_class, random_state=42)
    else:
        sampled_group = group.copy()
        missing_samples = samples_per_class - len(group)
        if not class_0.empty and missing_samples > 0:
           extra_samples = resample(class_0, n_samples=missing_samples, random_state=42, replace=True)
           sampled_group = pd.concat([sampled_group, extra_samples])
    balanced_data.append(sampled_group)

# sampling
balanced_samples = pd.concat(balanced_data)
# after sampling
actual_size = len(balanced_samples)
print(f"Actual total dataset size: {actual_size}")
print("Class distribution after sampling:")
print(balanced_samples[class_column].value_counts())


train_data, test_data = train_test_split(balanced_samples, test_size=0.3, random_state=define_random_state)

X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
X_val = test_data.iloc[:, :-1]
y_val = test_data.iloc[:, -1]

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_val, label=y_val)
watchlist = [(dtrain, 'train'), (dtest, 'eval')]


# Initialize parameters
iteration_count = 0
current_state = initial_state
policy_stable = False

improvement_history = []
best_f1_score = -np.inf
best_model = None
best_state = None

convergence_threshold = 1e-10  # Minimum improvement required
visited_states = set()
state_f1_cache = {}

neighbor_step = 1  # Initial neighbor step size


def get_neighbors(state, step_input):
    n, max_depth = state
    neighbors = []
    for step in range(1,step_input+1):
        if n - step >= n_values_min:
            neighbors.append((n - step, max_depth))
        if n + step <= n_values_max:
            neighbors.append((n + step, max_depth))
        if max_depth - step >= max_depth_values_min:
            neighbors.append((n, max_depth - step))
        if max_depth + step <= max_depth_values_max:
            neighbors.append((n, max_depth + step))
    return neighbors


def get_early_stopping_rounds(num_boost_round):
    #Calculate early_stopping_rounds is 10-20% of num_boost_round """
    lower_bound = int(num_boost_round * 0.1)  # 10% of num_boost_round
    # upper_bound = int(num_boost_round * 0.2)  # 20% of num_boost_round
    # return (lower_bound, upper_bound)
    return lower_bound

def evaluate_model(n, max_depth):
    if (n, max_depth) in state_f1_cache:
        # print(f"state({n}, {max_depth}) cache hit")
        return state_f1_cache[(n, max_depth)]

    start_time = time.time()
    params = {
        'max_depth': max_depth,
        'eta': 0.1,
        'objective': 'multi:softprob',  # to multi class
        'num_class': 13,
        'eval_metric': ['mlogloss', 'merror'],  # metrics for multi-class
        'tree_method': 'hist',
        #'tree_method': 'gpu_hist', #swich between GPU and CPU
        #'gpu_id': 0,
        #'max_bin': 256,
        'random_state': define_random_state
    }
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n,
        evals=watchlist,
        early_stopping_rounds=get_early_stopping_rounds(n),
        verbose_eval=50
    )

    f1_scores = []
    iloc_times = []
    sample_times = []
    train_times = []
    total_times = []
    train_time = time.time() - start_time

    start_time = time.time()
    predictions_prob = model.predict(dtest)  # probability of each class
    predictions = np.argmax(predictions_prob, axis=1)  # convert to class predictions
    total_time = time.time() - start_time + train_time
    f1 = f1_score(y_val, predictions, average='macro')
 
    state_f1_cache[(n, max_depth)] = (f1, train_time, total_time)
    # print(f"state({n}, {max_depth}): f1={f1}, train_time={train_time}, total_time={total_time}")
    return f1, train_time, total_time


stop_reason = ""
start_time_limit = time.time()
while True :
    iteration_count += 1
    # print(f"Iteration: {iteration_count}")
    training_time_used = time.time() - start_time_limit
    # print(f"Iteration: {iteration_count}, training_time_used: {training_time_used}")
    if training_time_used >= max_time_limit:
        print(f"training_time_used: {training_time_used} exceed max_time_limit: {max_time_limit}")
        break
    if iteration_count >= max_iterations:
        print(f"Reaching max iteration: {max_iterations}")
        break    
    if len(state_f1_cache) >= max_states:
        print(f"Max {max_states} stages reached")
        break
    elif  len(state_f1_cache) >= total_states:
        print(f"All {total_states} states visited")
        break
    policy_stable = True
    neighbors = get_neighbors(current_state, neighbor_step)
    # print(f"Neighbors: {neighbors}")
    current_n, current_max_depth = current_state
    current_f1_score, current_train_time, current_total_time = evaluate_model(current_n, current_max_depth)

    for neighbor in neighbors:
        if neighbor in visited_states:
            # print(f"neighbor: {neighbor} has been visited, skip this state")
            continue

        neighbor_n, neighbor_max_depth = neighbor
        neighbor_f1_score, _, _ = evaluate_model(neighbor_n, neighbor_max_depth)
        # print(f"neighbor: {neighbor}, neighbor_f1_score: {neighbor_f1_score:.15f}")

        if neighbor_f1_score > current_f1_score + convergence_threshold:
            current_state = neighbor
            current_f1_score = neighbor_f1_score
            policy_stable = False
            break
  
    # print(f"current_state: {current_state}, current_f1_score: {current_f1_score}, current_train_time: {current_train_time}, current_total_time: {current_total_time}")
    visited_states.add(current_state)

    if current_f1_score > best_f1_score:
        best_f1_score = current_f1_score
        best_model_n_estimator, best_model_max_depth = current_state
        # best_model = RandomForestClassifier(n_estimators=current_state[0], max_depth=current_state[1])
        best_state = current_state
        improvement_history.append((current_state, current_f1_score, current_train_time, current_total_time, iteration_count, training_time_used, neighbor_step, len(state_f1_cache)))
        print(f"iteration: {iteration_count}, training_time_used: {training_time_used}, neighbor_step now: {neighbor_step}, state_visited: {len(state_f1_cache)}")
        print(f"best_state: {current_state}, current_f1_score: {current_f1_score}, current_train_time: {current_train_time}, current_total_time: {current_total_time}")
    else:
        neighbor_step += 1  # Increase the neighbor step size
        # print(f"neighbor_step now: {neighbor_step}")

    policy_stable = False

# Display stop reason
print(f"\nTraining stopped: {stop_reason}")

# Display the F1-scores and states from history
print("Improvement History:")
for state, f1, train_time, total_time, iteration_count, training_time_used, neighbor_step, state_visited in improvement_history:
    print(f"State: n={state[0]}, max_depth={state[1]}, F1-Score: {f1:.15f}, Train Time: {train_time:.5f}s, Total Time: {total_time:.5f}s, iteration: {iteration_count}, training_time_used: {training_time_used}, neighbor_step now: {neighbor_step}, state_visited: {state_visited}")

# Display the best model
print("\nBest Model (from samples):")
print(f"State: n={best_state[0]}, max_depth={best_state[1]}, F1-Score: {best_f1_score:.15f}")



# Start cross validation
print(f"### Start cross validation (5 splits) on full dataset")
start_time = time.time()
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
print(f"iloc_time = {time.time() - start_time}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=define_random_state)
k = skf.split(X, y)

f1_scores = []
recall_per_fold = []
all_tp = []
all_tn = []
all_fp = []
all_fn = []

for fold_idx, (train_index, val_index) in enumerate(k):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_val, label=y_val)
    
    params = {
        'max_depth': best_model_max_depth,
        'eta': 0.1,
        'objective': 'multi:softprob',
        'num_class': 13,
        'eval_metric': ['mlogloss', 'merror'],
        'tree_method': 'hist',
        'random_state': define_random_state
    }
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=best_model_n_estimator,
        evals=[(dtest, 'eval')],
        early_stopping_rounds=get_early_stopping_rounds(best_model_n_estimator),
        verbose_eval=50
    )
    
    predictions_prob = model.predict(dtest)
    predictions = np.argmax(predictions_prob, axis=1)
    
    f1 = f1_score(y_val, predictions, average='macro')
    f1_scores.append(f1)

    
    recall = recall_score(y_val, predictions, average=None)
    recall_per_fold.append(recall)
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, predictions)
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)
    
    # collect TP, TN, FP, FN and find average
    all_tp.append(TP)
    all_tn.append(TN)
    all_fp.append(FP)
    all_fn.append(FN)
    
    print(f"\nFold {fold_idx + 1}:")
    print(f"Recall per class: {recall}")
    print(f"TP: {TP}")
    print(f"TN: {TN}")
    print(f"FP: {FP}")
    print(f"FN: {FN}")



# average of TP, TN, FP, FN of each class
mean_tp = np.mean(np.array(all_tp), axis=0)
mean_tn = np.mean(np.array(all_tn), axis=0)
mean_fp = np.mean(np.array(all_fp), axis=0)
mean_fn = np.mean(np.array(all_fn), axis=0)

mean_tp_overall = np.sum(mean_tp) * 5 / 13 / full_size
mean_tn_overall = np.sum(mean_tn) * 5 / 13 / full_size
mean_fp_overall = np.sum(mean_fp) * 5 / 13 / full_size
mean_fn_overall = np.sum(mean_fn) * 5 / 13 / full_size

# average F1-Score
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)
evaluation_time = time.time() - start_time

# results
print("\n=== Summary ===")
print(f"Evaluation time = {evaluation_time:.2f} seconds")
print(f"Mean F1-Score: {mean_f1:.15f} ± {std_f1:.15f}")

# mean of each class
print(f"\nMean TP per class: {mean_tp}")
print(f"Mean TN per class: {mean_tn}")
print(f"Mean FP per class: {mean_fp}")
print(f"Mean FN per class: {mean_fn}")

# mean of all class
print(f"\nOverall Mean TP: {mean_tp_overall:.15f}")
print(f"Overall Mean TN: {mean_tn_overall:.15f}")
print(f"Overall Mean FP: {mean_fp_overall:.15f}")
print(f"Overall Mean FN: {mean_fn_overall:.15f}")

# Best Model
print("\nBest Model (Cross validation - full dataset):")
print(f"State: n={best_state[0]}, max_depth={best_state[1]}, F1-Score: {mean_f1:.15f} ± {std_f1:.15f}, Cross-validation time: {evaluation_time:.2f} seconds")


# Save the best model
joblib.dump(best_model, "best_model.pkl")
print("Best model saved to 'best_model.pkl'")
