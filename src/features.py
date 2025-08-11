# src/features.py
import pandas as pd
import numpy as np

def rolling_feats(df: pd.DataFrame):
    df = df.sort_values(['player_id', 'week'])

    # Rolling recent form (no leakage: shift by 1)
    for stat in ['actual_points', 'proj_points']:
        df[f'{stat}_r3_mean'] = (
            df.groupby('player_id')[stat]
              .shift(1).rolling(3, min_periods=1).mean()
        )
        df[f'{stat}_r5_mean'] = (
            df.groupby('player_id')[stat]
              .shift(1).rolling(5, min_periods=1).mean()
        )
    df['actual_points_r3_med'] = (
        df.groupby('player_id')['actual_points']
          .shift(1).rolling(3, min_periods=1).median()
    )

    # Opponent strength vs position (allow fantasy points to POS, last 4 weeks)
    opp = (
        df[['week', 'opponent_team_id', 'pos', 'actual_points']]
        .dropna(subset=['actual_points'])
        .groupby(['opponent_team_id', 'pos', 'week'], as_index=False)['actual_points']
        .mean()
        .sort_values(['opponent_team_id', 'pos', 'week'])
    )
    opp['opp_pos_pts_r4'] = (
        opp.groupby(['opponent_team_id', 'pos'])['actual_points']
           .rolling(4, min_periods=1).mean()
           .reset_index(level=[0,1], drop=True)
    )
    df = df.merge(
        opp[['opponent_team_id', 'pos', 'week', 'opp_pos_pts_r4']],
        on=['opponent_team_id', 'pos', 'week'],
        how='left'
    )

    # Position flags
    df['is_qb'] = (df['pos'] == 'QB').astype(int)
    df['is_rb'] = (df['pos'] == 'RB').astype(int)
    df['is_wr'] = (df['pos'] == 'WR').astype(int)
    df['is_te'] = (df['pos'] == 'TE').astype(int)
    return df

def make_train(df):
    df = rolling_feats(df)
    train = df.dropna(subset=['actual_points']).copy()
    features = [
        'proj_points','proj_points_r3_mean','proj_points_r5_mean',
        'actual_points_r3_mean','actual_points_r5_mean','actual_points_r3_med',
        'opp_pos_pts_r4','is_qb','is_rb','is_wr','is_te'
    ]
    train = train.dropna(subset=features)
    X = train[features]
    y = train['actual_points']
    return X, y, features

def make_predict(df, target_week):
    df = rolling_feats(df)
    return df[df['week'] == target_week].copy()
