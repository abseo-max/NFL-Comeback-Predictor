import nflreadpy as nfl
import pandas as pd
import sklearn as sk
pbp_polars = nfl.load_pbp([2025,2024])
pbp = pbp_polars.to_pandas()                    
passes = pbp[pbp['qb_dropback'] == 1].copy()
cols = [
    # identifiers
    'game_id', 'season', 'week', 'season_type',

    # situation
    'qtr', 'game_seconds_remaining', 'score_differential',
    'posteam', 'defteam', 'down', 'ydstogo', 'yardline_100',

    # QB identity
    'passer_player_id', 'passer_player_name',

    # play outcome
    'yards_gained', 'air_yards', 'yards_after_catch',
    'epa', 'qb_epa', 'wpa',
    'pass_touchdown', 'complete_pass', 'incomplete_pass',
    'interception', 'sack', 'first_down',
    'cpoe', 'cp',

    # game result
    'result', 'home_team', 'away_team',
    'home_score', 'away_score',
]
passes = passes[cols].copy()
passes = passes.dropna(subset=['passer_player_id', 'passer_player_name'])
passes = passes.dropna(subset=['down'])

passes['air_yards'] = passes['air_yards'].fillna(0)
passes['yards_after_catch'] = passes['yards_after_catch'].fillna(0)
passes['cpoe'] = passes['cpoe'].fillna(0)
passes['cp'] = passes['cp'].fillna(0)

qb_per_game = (
    passes
    .groupby(['game_id', 'season', 'week', 'posteam', 
              'passer_player_id', 'passer_player_name'])
    .size()                          # count dropbacks per QB per game
    .reset_index(name='n_passes') # make it a normal dataframe
    .sort_values('n_passes',      # sort biggest first
                 ascending=False)
    .groupby(['game_id', 'season',   # per game per team...
              'posteam'])
    .first()                         # ...keep only the top QB
    .reset_index()                   # flatten again
)
starts = (
    qb_per_game.groupby(['passer_player_name', 'passer_player_id', 'season'])
    .size()
    .reset_index(name='starts')
)
total_starts = (
    starts.groupby(['passer_player_name', 'passer_player_id'])['starts']
    .sum()
    .reset_index(name='total_starts')
)
starter_threshold = 20

# get just the qualifying QB ids
starter_ids = total_starts[
    total_starts['total_starts'] >= starter_threshold
]['passer_player_id'].tolist()

# filter dropbacks to only those QBs
passes = passes[
    passes['passer_player_id'].isin(starter_ids)
].copy()

# get all 4th quarter plays for our QBs
q4 = passes[passes['qtr'] == 4].copy()

# find games where the QB's team was down 8+ at any point in Q4
comeback_opps = (
    q4[q4['score_differential'] <= -8]
    .groupby(['game_id', 'season', 'passer_player_name', 'passer_player_id', 'posteam'])
    .size()
    .reset_index(name='plays_while_down')
)

# get one row per game with the final result
game_results = (
    passes.groupby(['game_id', 'posteam'])['result']
    .first()
    .reset_index()
)

# merge result onto our comeback opportunities
comeback_opps = comeback_opps.merge(
    game_results,
    on=['game_id', 'posteam'],
    how='left'
)

# label it — did they win? 1 = yes, 0 = no
comeback_opps['came_back'] = (comeback_opps['result'] > 0).astype(int)

opp_keys = comeback_opps[['game_id', 'posteam']].copy()

# filter dropbacks to only those games, only Q4, only while down 8+
q4_down = passes.merge(opp_keys, on=['game_id', 'posteam'], how='inner')
q4_down = q4_down[
    (q4_down['qtr'] == 4) &
    (q4_down['score_differential'] <= -8)
].copy()

# now build features per game per QB
features = (
    q4_down.groupby(['game_id', 'passer_player_id', 'passer_player_name', 'posteam'])
    .agg(
        n_dropbacks      = ('epa', 'count'),
        epa_per_play     = ('epa', 'mean'),
        qb_epa_per_play  = ('qb_epa', 'mean'),
        comp_pct         = ('complete_pass', 'mean'),
        cpoe_avg         = ('cpoe', 'mean'),
        avg_air_yards    = ('air_yards', 'mean'),
        int_rate         = ('interception', 'mean'),
        sack_rate        = ('sack', 'mean'),
        td_rate          = ('pass_touchdown', 'mean'),
        first_down_rate  = ('first_down', 'mean'),
        yards_per_play   = ('yards_gained', 'mean'),
    )
    .reset_index()
)

model_data = features.merge(
    comeback_opps[['game_id', 'posteam', 'came_back']],
    on=['game_id', 'posteam'],
    how='left'
)

print(model_data.shape)
print(model_data['came_back'].value_counts())
print(model_data.isnull().sum())

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# define features and label
feature_cols = [
    'n_dropbacks', 'epa_per_play', 'qb_epa_per_play',
    'wpa_total', 'comp_pct', 'cpoe_avg', 'avg_air_yards',
    'int_rate', 'sack_rate', 'td_rate', 'first_down_rate',
    'yards_per_play'
]

X = model_data[feature_cols]
y = model_data['came_back']

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# evaluate
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))










