"""
FIFA World Cup 2026 AI Prediction Model v2
==========================================
Extensions:
  1. FIFA World Rankings as a feature
  2. Knockout stage simulator
  3. Parlay optimizer
"""

import os, urllib.request
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ── FIFA WORLD RANKINGS (June 2026 approximate) ───────────────────────────────
FIFA_RANKINGS = {
    'Argentina': 1, 'France': 2, 'Spain': 3, 'England': 4, 'Brazil': 5,
    'Portugal': 6, 'Netherlands': 7, 'Belgium': 8, 'Germany': 9, 'Morocco': 10,
    'Japan': 11, 'Colombia': 12, 'Uruguay': 13, 'Croatia': 14, 'United States': 15,
    'Mexico': 16, 'Senegal': 17, 'Ecuador': 18, 'Turkey': 19, 'Denmark': 20,
    'Switzerland': 21, 'South Korea': 22, 'Australia': 23, 'Norway': 24,
    'Canada': 25, 'Tunisia': 26, 'Algeria': 27, 'Egypt': 28, 'Saudi Arabia': 29,
    'Sweden': 30, 'Austria': 31, 'New Zealand': 32, 'Panama': 33, 'Iran': 34,
    'Scotland': 35, 'South Africa': 36, 'Ghana': 37, 'Czech Republic': 38,
    'Ivory Coast': 39, 'Jordan': 40, 'DR Congo': 41, 'Paraguay': 42,
    'Cape Verde': 43, 'Qatar': 44, 'Haiti': 45, 'Uzbekistan': 46,
    'Iraq': 47, 'Bolivia': 48, 'Curaçao': 49, 'Bosnia and Herzegovina': 50,
}

def get_rank(team):
    return FIFA_RANKINGS.get(team, 60)

# ── WC 2026 GROUP STAGE FIXTURES ─────────────────────────────────────────────
WC2026_GROUPS = {
    'A': ['Canada', 'Bosnia and Herzegovina', 'Switzerland', 'Qatar'],
    'B': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['Germany', 'Ivory Coast', 'Ecuador', 'Curaçao'],
    'E': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'F': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'G': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'H': ['France', 'Senegal', 'Iraq', 'Norway'],
    'I': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'J': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'K': ['England', 'Croatia', 'Ghana', 'Panama'],
    'L': ['Czech Republic', 'South Africa', 'Mexico', 'South Korea'],
}

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("=" * 65)
print("  FIFA WORLD CUP 2026 AI MODEL v2")
print("  With Rankings + Knockout Simulator + Parlay Optimizer")
print("=" * 65)

DATA_FILE = 'all_international_results.csv'
if not os.path.exists(DATA_FILE):
    print("📥 Downloading match data...")
    url = 'https://raw.githubusercontent.com/martj42/international_results/master/results.csv'
    urllib.request.urlretrieve(url, DATA_FILE)
    print("✅ Data downloaded!\n")

df = pd.read_csv(DATA_FILE)
df['date'] = pd.to_datetime(df['date'])

wc_finals   = df[df['tournament'] == 'FIFA World Cup'].copy()
competitive = df[df['tournament'].str.contains(
    'FIFA|UEFA|CAF|CONMEBOL|AFC|CONCACAF|Copa|Euro|Nations', na=False
)].copy()

train_df = competitive[
    (competitive['date'] >= '1990-01-01') &
    (competitive['home_score'].notna())
].copy().reset_index(drop=True)

print(f"\n✅ Training data: {len(train_df):,} matches")

# ── 2. FEATURE ENGINEERING (with FIFA rankings) ───────────────────────────────
def compute_team_stats(data, team, date, window=30):
    past = data[
        ((data['home_team'] == team) | (data['away_team'] == team)) &
        (data['date'] < date) &
        (data['home_score'].notna())
    ].tail(window)

    if len(past) == 0:
        return {'goals_scored': 1.2, 'goals_conceded': 1.0,
                'win_rate': 0.4, 'draw_rate': 0.25, 'form_pts': 1.2}

    goals_scored, goals_conceded, wins, draws = [], [], 0, 0
    for _, row in past.iterrows():
        if row['home_team'] == team:
            goals_scored.append(float(row['home_score']))
            goals_conceded.append(float(row['away_score']))
            if row['home_score'] > row['away_score']:    wins += 1
            elif row['home_score'] == row['away_score']: draws += 1
        else:
            goals_scored.append(float(row['away_score']))
            goals_conceded.append(float(row['home_score']))
            if row['away_score'] > row['home_score']:    wins += 1
            elif row['away_score'] == row['home_score']: draws += 1

    n = len(past)
    recent = past.tail(10)
    recent_pts = 0
    for _, row in recent.iterrows():
        if row['home_team'] == team:
            if row['home_score'] > row['away_score']:    recent_pts += 3
            elif row['home_score'] == row['away_score']: recent_pts += 1
        else:
            if row['away_score'] > row['home_score']:    recent_pts += 3
            elif row['away_score'] == row['home_score']: recent_pts += 1

    return {
        'goals_scored':   float(np.mean(goals_scored)),
        'goals_conceded': float(np.mean(goals_conceded)),
        'win_rate':       float(wins / n),
        'draw_rate':      float(draws / n),
        'form_pts':       float(recent_pts / max(len(recent), 1)),
    }

def build_features(data, source_data):
    rows = []
    for _, row in data.iterrows():
        h = compute_team_stats(source_data, row['home_team'], row['date'])
        a = compute_team_stats(source_data, row['away_team'], row['date'])
        h_rank = get_rank(row['home_team'])
        a_rank = get_rank(row['away_team'])
        rows.append([
            h['goals_scored'],    h['goals_conceded'],  h['win_rate'],
            h['draw_rate'],       h['form_pts'],
            a['goals_scored'],    a['goals_conceded'],  a['win_rate'],
            a['draw_rate'],       a['form_pts'],
            h['goals_scored']   - a['goals_scored'],
            h['goals_conceded'] - a['goals_conceded'],
            h['win_rate']       - a['win_rate'],
            h['form_pts']       - a['form_pts'],
            float(h_rank),        float(a_rank),
            float(a_rank - h_rank),           # positive = home is better ranked
            float(row.get('neutral', False)),
        ])
    return np.array(rows, dtype=np.float64)

def get_result(row):
    if row['home_score'] > row['away_score']:  return 'H'
    elif row['home_score'] < row['away_score']: return 'A'
    else:                                        return 'D'

train_df['result'] = train_df.apply(get_result, axis=1)

print("⚙️  Engineering features (including FIFA rankings)...")
X        = build_features(train_df, train_df)
y_result = np.array(train_df['result'].tolist())
y_home   = train_df['home_score'].to_numpy(dtype=float).astype(int)
y_away   = train_df['away_score'].to_numpy(dtype=float).astype(int)

print(f"   Feature matrix: {X.shape} (18 features incl. FIFA rankings)")

# ── 3. TRAIN MODELS ───────────────────────────────────────────────────────────
X_tr, X_te, yr_tr, yr_te = train_test_split(X, y_result, test_size=0.2, random_state=42)
_,    _,    yh_tr, yh_te = train_test_split(X, y_home,   test_size=0.2, random_state=42)
_,    _,    ya_tr, ya_te = train_test_split(X, y_away,   test_size=0.2, random_state=42)

print("\n🤖 Training models...")
clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
clf.fit(X_tr, yr_tr)
acc = accuracy_score(yr_te, clf.predict(X_te))
print(f"   ✅ Result classifier accuracy: {acc:.1%}")

reg_home = RandomForestRegressor(n_estimators=150, random_state=42)
reg_away = RandomForestRegressor(n_estimators=150, random_state=42)
reg_home.fit(X_tr, yh_tr)
reg_away.fit(X_tr, ya_tr)

classes = list(clf.classes_)

# ── PREDICT SINGLE MATCH ──────────────────────────────────────────────────────
def predict_match(home, away, neutral=True):
    ref_date = pd.Timestamp('2026-06-01')
    h = compute_team_stats(train_df, home, ref_date)
    a = compute_team_stats(train_df, away, ref_date)
    h_rank = get_rank(home)
    a_rank = get_rank(away)
    feat = np.array([[
        h['goals_scored'],    h['goals_conceded'],  h['win_rate'],
        h['draw_rate'],       h['form_pts'],
        a['goals_scored'],    a['goals_conceded'],  a['win_rate'],
        a['draw_rate'],       a['form_pts'],
        h['goals_scored']   - a['goals_scored'],
        h['goals_conceded'] - a['goals_conceded'],
        h['win_rate']       - a['win_rate'],
        h['form_pts']       - a['form_pts'],
        float(h_rank),        float(a_rank),
        float(a_rank - h_rank),
        float(neutral),
    ]], dtype=np.float64)
    proba = clf.predict_proba(feat)[0]
    probs = dict(zip(classes, proba))
    ph = probs.get('H', 0)
    pd_ = probs.get('D', 0)
    pa = probs.get('A', 0)
    gh = int(np.clip(np.round(reg_home.predict(feat)[0]), 0, 6))
    ga = int(np.clip(np.round(reg_away.predict(feat)[0]), 0, 6))
    return {'home': home, 'away': away, 'h_win': ph, 'draw': pd_, 'a_win': pa,
            'pred_home_goals': gh, 'pred_away_goals': ga,
            'confidence': max(ph, pd_, pa)}

# ── 4. GROUP STAGE SIMULATION ─────────────────────────────────────────────────
print("\n\n" + "=" * 65)
print("  📊 GROUP STAGE SIMULATION")
print("=" * 65)

group_standings = {}
group_match_results = {}

for group, teams in WC2026_GROUPS.items():
    standings = {t: {'pts': 0, 'gf': 0, 'ga': 0, 'gd': 0, 'w': 0, 'd': 0, 'l': 0} for t in teams}
    matches = []
    for home, away in combinations(teams, 2):
        m = predict_match(home, away, neutral=True)
        # Determine match outcome for points
        if m['h_win'] > m['a_win'] and m['h_win'] > m['draw']:
            # Home win
            standings[home]['pts'] += 3
            standings[home]['w']   += 1
            standings[away]['l']   += 1
            result = f"{home} Win"
        elif m['a_win'] > m['h_win'] and m['a_win'] > m['draw']:
            # Away win
            standings[away]['pts'] += 3
            standings[away]['w']   += 1
            standings[home]['l']   += 1
            result = f"{away} Win"
        else:
            # Draw
            standings[home]['pts'] += 1
            standings[away]['pts'] += 1
            standings[home]['d']   += 1
            standings[away]['d']   += 1
            result = "Draw"

        standings[home]['gf'] += m['pred_home_goals']
        standings[home]['ga'] += m['pred_away_goals']
        standings[away]['gf'] += m['pred_away_goals']
        standings[away]['ga'] += m['pred_home_goals']
        standings[home]['gd'] = standings[home]['gf'] - standings[home]['ga']
        standings[away]['gd'] = standings[away]['gf'] - standings[away]['ga']

        matches.append({
            'match': f"{home} vs {away}",
            'result': result,
            'score': f"{m['pred_home_goals']} - {m['pred_away_goals']}",
            'h_win%': f"{m['h_win']:.0%}",
            'draw%':  f"{m['draw']:.0%}",
            'a_win%': f"{m['a_win']:.0%}",
        })

    sorted_teams = sorted(standings.items(), key=lambda x: (x[1]['pts'], x[1]['gd'], x[1]['gf']), reverse=True)
    group_standings[group] = sorted_teams
    group_match_results[group] = matches

    print(f"\n  GROUP {group}:")
    print(f"  {'Team':<30} {'Pts':>4} {'W':>3} {'D':>3} {'L':>3} {'GF':>3} {'GA':>3} {'GD':>4}")
    print(f"  {'-'*55}")
    for i, (team, s) in enumerate(sorted_teams):
        arrow = " ✅" if i < 2 else " ❌"
        print(f"  {team:<30} {s['pts']:>4} {s['w']:>3} {s['d']:>3} {s['l']:>3} {s['gf']:>3} {s['ga']:>3} {s['gd']:>4}{arrow}")

# Extract group winners and runners-up
qualifiers = {}
for group, standings in group_standings.items():
    qualifiers[f"{group}1"] = standings[0][0]  # winner
    qualifiers[f"{group}2"] = standings[1][0]  # runner-up

# ── 5. KNOCKOUT STAGE SIMULATOR ───────────────────────────────────────────────
print("\n\n" + "=" * 65)
print("  🏆 KNOCKOUT STAGE SIMULATOR")
print("=" * 65)

# Round of 32 (48 teams → 24 groups, top 2 + 8 best 3rd place → 32)
# Simplified: use group winners vs runners-up from adjacent groups
# WC 2026 has 12 groups, 48 teams, top 2 from each = 24, + 8 best 3rd = 32 total
# For simplicity: simulate R32 as A1vsB2, B1vsA2, C1vsD2, etc.

round_of_32_matchups = [
    (qualifiers['A1'], qualifiers['B2']),
    (qualifiers['B1'], qualifiers['A2']),
    (qualifiers['C1'], qualifiers['D2']),
    (qualifiers['D1'], qualifiers['C2']),
    (qualifiers['E1'], qualifiers['F2']),
    (qualifiers['F1'], qualifiers['E2']),
    (qualifiers['G1'], qualifiers['H2']),
    (qualifiers['H1'], qualifiers['G2']),
    (qualifiers['I1'], qualifiers['J2']),
    (qualifiers['J1'], qualifiers['I2']),
    (qualifiers['K1'], qualifiers['L2']),
    (qualifiers['L1'], qualifiers['K2']),
    # 8 best 3rd place teams (simplified as next group runners-up)
    (qualifiers['A1'], qualifiers['C2']),
    (qualifiers['B1'], qualifiers['D2']),
    (qualifiers['E1'], qualifiers['G2']),
    (qualifiers['F1'], qualifiers['H2']),
]

def simulate_knockout_round(matchups, round_name):
    print(f"\n  {round_name}:")
    print(f"  {'Match':<45} {'Score':>7}  {'Winner':<30} {'Conf':>5}")
    print(f"  {'-'*90}")
    winners = []
    for home, away in matchups:
        m = predict_match(home, away, neutral=True)
        # In knockout, no draws — winner is higher probability side
        if m['h_win'] >= m['a_win']:
            winner = home
            conf = m['h_win'] / (m['h_win'] + m['a_win'])
        else:
            winner = away
            conf = m['a_win'] / (m['h_win'] + m['a_win'])
        score = f"{m['pred_home_goals']} - {m['pred_away_goals']}"
        match_str = f"{home} vs {away}"
        print(f"  {match_str:<45} {score:>7}  {winner:<30} {conf:.0%}")
        winners.append(winner)
    return winners

r32_winners    = simulate_knockout_round(round_of_32_matchups, "ROUND OF 32")
r16_matchups   = [(r32_winners[i], r32_winners[i+1]) for i in range(0, len(r32_winners), 2)]
r16_winners    = simulate_knockout_round(r16_matchups, "ROUND OF 16")
qf_matchups    = [(r16_winners[i], r16_winners[i+1]) for i in range(0, len(r16_winners), 2)]
qf_winners     = simulate_knockout_round(qf_matchups, "QUARTER FINALS")
sf_matchups    = [(qf_winners[i], qf_winners[i+1]) for i in range(0, len(qf_winners), 2)]
sf_winners     = simulate_knockout_round(sf_matchups, "SEMI FINALS")
final_matchups = [(sf_winners[i], sf_winners[i+1]) for i in range(0, len(sf_winners), 2)]
final_winners  = simulate_knockout_round(final_matchups, "🏆 WORLD CUP FINAL")

print(f"\n  🏆 WORLD CUP 2026 CHAMPION (AI prediction): {final_winners[0]}")

# ── 6. PARLAY OPTIMIZER ───────────────────────────────────────────────────────
print("\n\n" + "=" * 65)
print("  💰 PARLAY OPTIMIZER")
print("=" * 65)

# Load group stage predictions
wc_2026 = wc_finals[wc_finals['home_score'].isna()].copy().reset_index(drop=True)
all_preds = []
for _, row in wc_2026.iterrows():
    m = predict_match(row['home_team'], row['away_team'], neutral=True)
    best_outcome = max([('H', m['h_win']), ('D', m['draw']), ('A', m['a_win'])], key=lambda x: x[1])
    outcome_label = f"{row['home_team']} Win" if best_outcome[0]=='H' else \
                    f"{row['away_team']} Win" if best_outcome[0]=='A' else "Draw"
    fair_odds = round(1 / best_outcome[1], 2) if best_outcome[1] > 0.01 else 99.0
    all_preds.append({
        'match':        f"{row['home_team']} vs {row['away_team']}",
        'date':         row['date'].strftime('%d %b'),
        'pick':         outcome_label,
        'prob':         best_outcome[1],
        'fair_odds':    fair_odds,
        'conf':         best_outcome[1],
    })

preds_sorted = sorted(all_preds, key=lambda x: x['prob'], reverse=True)

print("\n  📌 BEST SINGLE BETS (by confidence):")
print(f"  {'Match':<40} {'Pick':<25} {'Prob':>5}  {'Fair Odds':>9}")
print(f"  {'-'*85}")
for p in preds_sorted[:10]:
    print(f"  {p['match']:<40} {p['pick']:<25} {p['prob']:>5.0%}  {p['fair_odds']:>9.2f}")

# 2-leg parlays
print("\n  🎯 BEST 2-LEG PARLAYS (combined probability):")
print(f"  {'Leg 1':<32} {'Leg 2':<32} {'Combined Prob':>14} {'Combined Odds':>14}")
print(f"  {'-'*95}")
top15 = preds_sorted[:15]
parlays_2 = []
for a, b in combinations(top15, 2):
    combined_prob = a['prob'] * b['prob']
    combined_odds = round(a['fair_odds'] * b['fair_odds'], 2)
    parlays_2.append((a, b, combined_prob, combined_odds))
parlays_2.sort(key=lambda x: x[2], reverse=True)
for a, b, cp, co in parlays_2[:8]:
    print(f"  {a['pick']:<32} {b['pick']:<32} {cp:>14.0%} {co:>14.2f}")

# 3-leg parlays
print("\n  🔥 BEST 3-LEG PARLAYS (best risk/reward):")
print(f"  {'Picks':<75} {'Prob':>6} {'Odds':>7}")
print(f"  {'-'*92}")
top10 = preds_sorted[:10]
parlays_3 = []
for a, b, c in combinations(top10, 3):
    combined_prob = a['prob'] * b['prob'] * c['prob']
    combined_odds = round(a['fair_odds'] * b['fair_odds'] * c['fair_odds'], 2)
    parlays_3.append((a, b, c, combined_prob, combined_odds))
parlays_3.sort(key=lambda x: x[3], reverse=True)
for a, b, c, cp, co in parlays_3[:6]:
    picks = f"{a['pick']} + {b['pick']} + {c['pick']}"
    print(f"  {picks:<75} {cp:>6.0%} {co:>7.2f}")

print("\n  ⚠️  Disclaimer: For educational purposes. Bet responsibly.")
print("\n✅ Done! Check wc2026_predictions.csv for all group stage predictions.")
print("=" * 65)
