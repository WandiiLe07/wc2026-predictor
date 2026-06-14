# ⚽ FIFA World Cup 2026 — AI Prediction Model

A machine learning system that predicts match outcomes, scorelines, and generates betting insights for all FIFA World Cup 2026 fixtures.

Built by **Mlungisi Wandile Luthuli** · BICIOT IoT Student @ Durban University of Technology

---

## 🚀 Features

- **Match Outcome Prediction** — Gradient Boosting classifier trained on 16,054 competitive international matches (1990–2026)
- **Scoreline Prediction** — Random Forest regressors predict home and away goals
- **FIFA Rankings Integration** — Real June 2026 rankings used as model features
- **Group Stage Simulation** — Full standings table for all 12 WC 2026 groups
- **Knockout Stage Simulator** — Simulates R32 → R16 → QF → SF → Final
- **Parlay Optimizer** — Best single, 2-leg and 3-leg parlay bets by combined probability
- **Player Intelligence Layer** — 46 key players analysed with impact ratings and underdog deep dives
- **Betting Insights** — Fair odds, value bets, draw specialists and goal predictions

---

## 📁 Files

| File | Description |
|------|-------------|
| `wc_predictor.py` | Base ML model — group stage predictions |
| `wc_predictor_v2.py` | Full model — rankings + knockout simulator + parlay optimizer |
| `wc_player_analysis.py` | Player intelligence — 46 players, underdog analysis, golden boot |
| `wc2026_predictions.csv` | Raw predictions with probabilities and fair odds |
| `WC2026_AI_Prediction_Report.html` | Interactive HTML betting report (open in browser) |

---

## 🛠️ Setup

**Requirements:**
```bash
pip install pandas numpy scikit-learn
```

**Run base predictions:**
```bash
python wc_predictor.py
```

**Run full model (rankings + knockouts + parlays):**
```bash
python wc_predictor_v2.py
```

**Run player intelligence report:**
```bash
python wc_player_analysis.py
```

> The scripts automatically download the match data on first run. No manual setup needed.

---

## 🤖 Model Details

| Property | Value |
|----------|-------|
| Algorithm | Gradient Boosting Classifier (GBM) |
| Score predictor | Random Forest Regressor |
| Training data | 16,054 competitive international matches |
| Time range | 1990 – 2026 |
| Test accuracy | ~60% |
| Features | 18 (rolling stats, form, FIFA rankings, goal differentials) |

**Features used:**
- Rolling goals scored/conceded (last 30 matches)
- Win rate and draw rate
- Recency-weighted form points (last 10 matches)
- FIFA world ranking (home and away)
- Ranking differential
- Neutral ground flag
- Home/away stat differentials

---

## 🔥 Key Predictions

- **Japan** are model's biggest favourites in Group E (72% vs Sweden)
- **Morocco** predicted to beat Brazil in Group C
- **Norway** (Haaland + Ødegaard) flagged as most dangerous underdog
- **Algeria** given 46% chance to beat Argentina
- **Best draw value:** Norway vs Senegal (47% draw probability, fair odds 2.15)

---

## 🏆 Golden Boot Prediction

| Rank | Player | Team | Predicted Goals |
|------|--------|------|----------------|
| 1 | Erling Haaland | Norway | 6–8 |
| 2 | Kylian Mbappé | France | 5–7 |
| 3 | Lamine Yamal | Spain | 4–6 |
| 4 | Harry Kane | England | 4–6 |
| 5 | Vinicius Jr | Brazil | 4–6 |

---

## ⚠️ Disclaimer

This project is for **educational and analytical purposes only**. Sports betting carries financial risk. The model predictions are based on historical data and statistical patterns — they do not guarantee outcomes. Bet responsibly.

---

## 🧠 Tech Stack

- Python 3.14
- scikit-learn (GBM, Random Forest)
- pandas & numpy
- Real match data: [martj42/international_results](https://github.com/martj42/international_results)

---

*Built as a data science portfolio project combining IoT data analytics skills, machine learning, and sports intelligence.*
