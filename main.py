import nflreadpy as nfl
import pandas as pd
import sklearn as sk
import numpy as np

pbp_polars = nfl.load_pbp([2025,2024,2023,2022,2021, 2020])
pbp = pbp_polars.to_pandas() 
qb_counts = pbp[pbp['qb_dropback'] == 1].groupby(['game_id', 'posteam', 'passer_player_name']).size().reset_index(name='pass_count')

# Sort by pass_count descending and grab the top one (the primary starting/playing QB)
primary_qbs = qb_counts.sort_values('pass_count', ascending=False).groupby(['game_id', 'posteam']).first().reset_index()

# Rename columns to create a clean lookup dictionary dataframe
team_qb_map = primary_qbs[['game_id', 'posteam', 'passer_player_name']].rename(
    columns={'posteam': 'team', 'passer_player_name': 'team_qb'}
)
#qb 4th quarter EPA till the moment

#calcute 4th Quarter EPA

q4_epa = pbp[
    (pbp['qtr'] == 4) &
    (pbp['qb_dropback']== 1)
].copy()


q4_epasum = q4_epa.groupby(['passer_player_name','game_id'])['epa'].sum().reset_index()

# 3. Sort chronologically by QB and Game ID
qb_game_q4_epa = q4_epasum.sort_values(by=['passer_player_name', 'game_id'])

# 4. Calculate cumulative Q4 EPA *prior* to the current game
# We take the cumulative sum, and subtract the current game's EPA so it only represents history up to that point
qb_game_q4_epa['q4_epa_till_moment'] = (
    qb_game_q4_epa.groupby('passer_player_name')['epa'].cumsum() - qb_game_q4_epa['epa']
)

# 5. Keep only the columns needed for merging
qb_career_q4_epa = qb_game_q4_epa[['passer_player_name', 'game_id', 'q4_epa_till_moment']]

passes = pbp[
    (pbp['qtr'] == 4)
]
passes = passes.sort_values(by=['game_id','play_id'])
firstplay = passes.groupby(['game_id']).first().reset_index()

firstplay['deficit'] = (firstplay['defteam_score'] - firstplay['posteam_score']).abs()

comeback = firstplay[
    (firstplay['deficit'] >= 7) &
    (firstplay['deficit'] <= 21)
    ].copy()

comeback['losing_team'] =np.where(
    comeback['defteam_score'] < comeback['posteam_score'], 
    comeback['defteam'], 
    comeback['posteam']
)
comeback['winning_team'] = np.where(
    comeback['defteam_score'] < comeback['posteam_score'], 
    comeback['posteam'], 
    comeback['defteam']

)

comeback['winner'] = np.where(
    comeback['home_score'] > comeback['away_score'], comeback['home_team'],
    np.where(comeback['away_score']> comeback['home_score'], comeback['away_team'], "tie"  )
) 
comeback = comeback.merge(
    team_qb_map, 
    left_on=['game_id', 'losing_team'], 
    right_on=['game_id', 'team'], 
    how='left'
)

comeback = comeback.rename(columns={'team_qb': 'losing_qb'})

comeback = comeback.merge(
    qb_career_q4_epa,
    left_on=['losing_qb', 'game_id'], 
    right_on=['passer_player_name', 'game_id'], 
    how='left'
)

comeback['q4_epa_till_moment'] = comeback['q4_epa_till_moment'].fillna(0)


comeback['cameback'] = (comeback['losing_team'] == comeback['winner']).astype(int)
cols = ['game_id', 'losing_qb', 'deficit', 'winning_team','q4_epa_till_moment', 'cameback']

comeback = comeback[cols]


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Drop any rows with missing data (e.g., if a QB wasn't mapped)
ml_data = comeback.dropna().copy()

# 1. Define Features (X) and Target (y)
features = ['losing_qb', 'deficit', 'winning_team', 'q4_epa_till_moment']
X = ml_data[features].copy()
y = ml_data['cameback']

# 2. Convert text columns to 'category' type so XGBoost can process them natively
X['losing_qb'] = X['losing_qb'].astype('category')
X['winning_team'] = X['winning_team'].astype('category')

# 3. Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

neg_class_count = (y_train == 0).sum()
pos_class_count = (y_train == 1).sum()

# Calculate the scale_pos_weight
imbalanceweight = neg_class_count / pos_class_count
# 4. Initialize and train the XGBoost Classifier
# enable_categorical=True is required to let XGBoost handle the QBs and Teams
model = xgb.XGBClassifier(
    objective = 'binary:logistic',
    enable_categorical=True, 
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight = imbalanceweight
)

model.fit(X_train, y_train)

# 5. Make predictions and evaluate the model
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

import shap 
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)











