"""
FIFA World Cup 2026 — Player Intelligence Layer
================================================
Analyzes key players, impact ratings, and adjusts team predictions
based on star player presence (like Haaland for Norway)
"""

import os
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# PLAYER DATABASE — WC 2026 SQUADS (Key Players Only)
# Impact score: 1-10 (how much they elevate their team above average)
# ═══════════════════════════════════════════════════════════════════

PLAYERS = {

    # ── EUROPE ────────────────────────────────────────────────────
    'Erling Haaland': {
        'team': 'Norway', 'position': 'ST', 'age': 25,
        'club': 'Manchester City', 'impact': 9.5,
        'goals_per_game': 0.91, 'assists_per_game': 0.3,
        'rating': 9.3,
        'note': 'Generational striker. Norway\'s entire attack built around him. Can single-handedly win games.'
    },
    'Martin Ødegaard': {
        'team': 'Norway', 'position': 'CAM', 'age': 27,
        'club': 'Arsenal', 'impact': 8.5,
        'goals_per_game': 0.35, 'assists_per_game': 0.45,
        'rating': 8.8,
        'note': 'World-class playmaker. Creates chances at elite level. Together with Haaland, Norway is dangerous.'
    },
    'Kylian Mbappé': {
        'team': 'France', 'position': 'ST', 'age': 27,
        'club': 'Real Madrid', 'impact': 9.7,
        'goals_per_game': 0.85, 'assists_per_game': 0.4,
        'rating': 9.6,
        'note': 'Best player in the world. Pace, finishing, leadership. France\'s title threat runs through him.'
    },
    'Jude Bellingham': {
        'team': 'England', 'position': 'CM', 'age': 22,
        'club': 'Real Madrid', 'impact': 9.2,
        'goals_per_game': 0.42, 'assists_per_game': 0.38,
        'rating': 9.1,
        'note': 'Box-to-box excellence. Goals, assists, pressing. England\'s most complete player.'
    },
    'Bukayo Saka': {
        'team': 'England', 'position': 'RW', 'age': 24,
        'club': 'Arsenal', 'impact': 8.6,
        'goals_per_game': 0.38, 'assists_per_game': 0.42,
        'rating': 8.7,
        'note': 'Consistent, creative and direct. England\'s most reliable attacker.'
    },
    'Lionel Messi': {
        'team': 'Argentina', 'position': 'RW', 'age': 39,
        'club': 'Inter Miami', 'impact': 8.0,
        'goals_per_game': 0.55, 'assists_per_game': 0.6,
        'rating': 8.2,
        'note': 'GOAT. Age may be a factor but in big tournaments he always shows up. Will this be his last WC?'
    },
    'Lautaro Martínez': {
        'team': 'Argentina', 'position': 'ST', 'age': 27,
        'club': 'Inter Milan', 'impact': 8.8,
        'goals_per_game': 0.72, 'assists_per_game': 0.28,
        'rating': 8.9,
        'note': 'Lethal finisher. Argentina\'s main goal threat now. Clinical in front of goal.'
    },
    'Vinicius Jr': {
        'team': 'Brazil', 'position': 'LW', 'age': 25,
        'club': 'Real Madrid', 'impact': 9.4,
        'goals_per_game': 0.65, 'assists_per_game': 0.5,
        'rating': 9.3,
        'note': 'Electric. Best dribbler in the world. Brazil\'s most dangerous attacker.'
    },
    'Rodrygo': {
        'team': 'Brazil', 'position': 'RW', 'age': 24,
        'club': 'Real Madrid', 'impact': 8.3,
        'goals_per_game': 0.4, 'assists_per_game': 0.35,
        'rating': 8.4,
        'note': 'Big game player. Scores in knockout football. Partners brilliantly with Vinicius.'
    },
    'Cristiano Ronaldo': {
        'team': 'Portugal', 'position': 'ST', 'age': 41,
        'club': 'Al Nassr', 'impact': 7.2,
        'goals_per_game': 0.5, 'assists_per_game': 0.15,
        'rating': 7.5,
        'note': 'Legend but physical decline is real at 41. Portugal are stronger without him starting. Still dangerous from set pieces.'
    },
    'Bruno Fernandes': {
        'team': 'Portugal', 'position': 'CAM', 'age': 31,
        'club': 'Manchester United', 'impact': 8.8,
        'goals_per_game': 0.38, 'assists_per_game': 0.5,
        'rating': 8.7,
        'note': 'Portugal\'s real creative engine. Free kicks, vision, leadership. Carries them in big games.'
    },
    'Pedri': {
        'team': 'Spain', 'position': 'CM', 'age': 23,
        'club': 'Barcelona', 'impact': 8.9,
        'goals_per_game': 0.22, 'assists_per_game': 0.38,
        'rating': 8.8,
        'note': 'Generational midfielder. Controls tempo, presses relentlessly. Spain\'s heartbeat.'
    },
    'Lamine Yamal': {
        'team': 'Spain', 'position': 'RW', 'age': 18,
        'club': 'Barcelona', 'impact': 9.0,
        'goals_per_game': 0.45, 'assists_per_game': 0.55,
        'rating': 9.1,
        'note': 'TEENAGER but already world class. EURO 2024 winner. Will be the best player at this WC.'
    },
    'Florian Wirtz': {
        'team': 'Germany', 'position': 'CAM', 'age': 23,
        'club': 'Bayern Munich', 'impact': 9.1,
        'goals_per_game': 0.4, 'assists_per_game': 0.55,
        'rating': 9.0,
        'note': 'Exceptional talent. Germany\'s new Özil but more dynamic. Will light up the tournament.'
    },
    'Jamal Musiala': {
        'team': 'Germany', 'position': 'AM', 'age': 23,
        'club': 'Bayern Munich', 'impact': 8.9,
        'goals_per_game': 0.38, 'assists_per_game': 0.42,
        'rating': 8.8,
        'note': 'Silky, fast, clinical. With Wirtz, Germany have the most exciting midfield at the tournament.'
    },
    'Virgil van Dijk': {
        'team': 'Netherlands', 'position': 'CB', 'age': 35,
        'club': 'Liverpool', 'impact': 8.5,
        'goals_per_game': 0.12, 'assists_per_game': 0.08,
        'rating': 8.6,
        'note': 'Dominant defender. Makes Netherlands incredibly hard to beat. Leader on and off the pitch.'
    },
    'Cody Gakpo': {
        'team': 'Netherlands', 'position': 'LW', 'age': 25,
        'club': 'Liverpool', 'impact': 8.3,
        'goals_per_game': 0.45, 'assists_per_game': 0.3,
        'rating': 8.3,
        'note': 'Versatile attacker. Good tournament player. Liverpool sharpness showing.'
    },
    'Kevin De Bruyne': {
        'team': 'Belgium', 'position': 'CM', 'age': 35,
        'club': 'Manchester City', 'impact': 8.0,
        'goals_per_game': 0.2, 'assists_per_game': 0.55,
        'rating': 8.2,
        'note': 'Last WC likely. Still elite vision and passing but pace has dropped. Belgium\'s best player.'
    },
    'Romelu Lukaku': {
        'team': 'Belgium', 'position': 'ST', 'age': 33,
        'club': 'Napoli', 'impact': 7.8,
        'goals_per_game': 0.58, 'assists_per_game': 0.2,
        'rating': 7.9,
        'note': 'Physical presence upfront. Still dangerous in the box. Belgium need him fit.'
    },
    'Harry Kane': {
        'team': 'England', 'position': 'ST', 'age': 32,
        'club': 'Bayern Munich', 'impact': 9.0,
        'goals_per_game': 0.78, 'assists_per_game': 0.35,
        'rating': 9.0,
        'note': 'Elite goalscorer. Finally winning with Bayern. Hungry to win the World Cup after years of hurt.'
    },

    # ── AFRICA ────────────────────────────────────────────────────
    'Achraf Hakimi': {
        'team': 'Morocco', 'position': 'RB', 'age': 27,
        'club': 'PSG', 'impact': 8.7,
        'goals_per_game': 0.18, 'assists_per_game': 0.42,
        'rating': 8.8,
        'note': 'Best right back in the world. Attacking threat and defensive rock. Morocco\'s most important player.'
    },
    'Sofyan Amrabat': {
        'team': 'Morocco', 'position': 'CDM', 'age': 28,
        'club': 'Fiorentina', 'impact': 8.2,
        'goals_per_game': 0.05, 'assists_per_game': 0.12,
        'rating': 8.3,
        'note': 'Elite ball-winner. Morocco\'s defensive shield. Makes them almost impossible to break down.'
    },
    'Sadio Mané': {
        'team': 'Senegal', 'position': 'LW', 'age': 34,
        'club': 'Al Nassr', 'impact': 7.8,
        'goals_per_game': 0.4, 'assists_per_game': 0.25,
        'rating': 7.9,
        'note': 'Experienced leader. Slight decline but still clinical. Senegal go as far as Mané takes them.'
    },
    'Mohamed Salah': {
        'team': 'Egypt', 'position': 'RW', 'age': 34,
        'club': 'Liverpool', 'impact': 9.2,
        'goals_per_game': 0.68, 'assists_per_game': 0.45,
        'rating': 9.1,
        'note': 'Still elite at Liverpool. Egypt\'s entire attack runs through him. Can drag weak teams far.'
    },
    'Riyad Mahrez': {
        'team': 'Algeria', 'position': 'RW', 'age': 35,
        'club': 'Al Ahli', 'impact': 7.5,
        'goals_per_game': 0.3, 'assists_per_game': 0.35,
        'rating': 7.6,
        'note': 'Fading but still tricky. Algeria need him sharp to cause upsets. Model underrates Algeria partially due to Mahrez.'
    },
    'Victor Osimhen': {
        'team': None, 'position': 'ST',  # Nigeria not qualified — just reference
        'club': 'Galatasaray', 'impact': 9.0, 'rating': 9.0,
        'note': 'Not at WC 2026 (Nigeria failed to qualify). Huge absence for African football.'
    },

    # ── ASIA / OCEANIA ────────────────────────────────────────────
    'Son Heung-min': {
        'team': 'South Korea', 'position': 'LW', 'age': 34,
        'club': 'Tottenham', 'impact': 8.8,
        'goals_per_game': 0.55, 'assists_per_game': 0.35,
        'rating': 8.8,
        'note': 'Elite pace and finishing. South Korea\'s captain and talisman. Can cause anyone problems.'
    },
    'Lee Kang-in': {
        'team': 'South Korea', 'position': 'AM', 'age': 23,
        'club': 'PSG', 'impact': 8.0,
        'goals_per_game': 0.3, 'assists_per_game': 0.4,
        'rating': 8.0,
        'note': 'PSG-trained creativity. Great technical ability. Partners well with Son.'
    },
    'Takehiro Tomiyasu': {
        'team': 'Japan', 'position': 'RB', 'age': 26,
        'club': 'Arsenal', 'impact': 7.8,
        'goals_per_game': 0.05, 'assists_per_game': 0.15,
        'rating': 7.9,
        'note': 'Solid, versatile. Japan\'s defensive organisation is elite and he\'s central to it.'
    },
    'Kaoru Mitoma': {
        'team': 'Japan', 'position': 'LW', 'age': 27,
        'club': 'Brighton', 'impact': 8.5,
        'goals_per_game': 0.42, 'assists_per_game': 0.38,
        'rating': 8.5,
        'note': 'Unstoppable dribbler. Best Japanese player since Nakata. Will be a WC star.'
    },
    'Wataru Endo': {
        'team': 'Japan', 'position': 'CDM', 'age': 31,
        'club': 'Liverpool', 'impact': 7.9,
        'goals_per_game': 0.08, 'assists_per_game': 0.12,
        'rating': 8.0,
        'note': 'Liverpool\'s defensive midfielder. Screens the back four brilliantly for Japan.'
    },

    # ── AMERICAS ──────────────────────────────────────────────────
    'Christian Pulisic': {
        'team': 'United States', 'position': 'LW', 'age': 27,
        'club': 'AC Milan', 'impact': 8.5,
        'goals_per_game': 0.42, 'assists_per_game': 0.32,
        'rating': 8.5,
        'note': 'Captain America. Playing the best football of his career at Milan. USA\'s standout player.'
    },
    'Tyler Adams': {
        'team': 'United States', 'position': 'CDM', 'age': 26,
        'club': 'Bournemouth', 'impact': 7.8,
        'goals_per_game': 0.06, 'assists_per_game': 0.14,
        'rating': 7.9,
        'note': 'Tenacious, technical. USA\'s defensive engine. Holds the team together.'
    },
    'Alphonso Davies': {
        'team': 'Canada', 'position': 'LB', 'age': 25,
        'club': 'Bayern Munich', 'impact': 9.0,
        'goals_per_game': 0.2, 'assists_per_game': 0.45,
        'rating': 9.0,
        'note': 'One of the fastest players alive. World-class fullback. Canada\'s most dangerous attacker going forward.'
    },
    'Jonathan David': {
        'team': 'Canada', 'position': 'ST', 'age': 24,
        'club': 'Lille', 'impact': 8.7,
        'goals_per_game': 0.75, 'assists_per_game': 0.25,
        'rating': 8.7,
        'note': 'Prolific scorer in Ligue 1. Clinical finisher. With Davies behind him Canada are genuine threats.'
    },
    'Neymar Jr': {
        'team': 'Brazil', 'position': 'LW', 'age': 34,
        'club': 'Al Hilal', 'impact': 7.0,
        'goals_per_game': 0.45, 'assists_per_game': 0.5,
        'rating': 7.2,
        'note': 'Injury history is a massive concern. If fit, Brazil have 4 world-class attackers. If injured, big hole.'
    },
    'Federico Valverde': {
        'team': 'Uruguay', 'position': 'CM', 'age': 26,
        'club': 'Real Madrid', 'impact': 8.8,
        'goals_per_game': 0.25, 'assists_per_game': 0.3,
        'rating': 8.8,
        'note': 'Box-to-box powerhouse. One of the best midfielders in the world. Drives Uruguay forward tirelessly.'
    },
    'Darwin Núñez': {
        'team': 'Uruguay', 'position': 'ST', 'age': 25,
        'club': 'Liverpool', 'impact': 8.3,
        'goals_per_game': 0.55, 'assists_per_game': 0.22,
        'rating': 8.3,
        'note': 'Explosive, powerful striker. Inconsistent but devastating on his day. Uruguay\'s goal threat.'
    },
    'Luis Díaz': {
        'team': 'Colombia', 'position': 'LW', 'age': 27,
        'club': 'Liverpool', 'impact': 8.7,
        'goals_per_game': 0.45, 'assists_per_game': 0.38,
        'rating': 8.7,
        'note': 'Electric winger. Best season at Liverpool. Colombia\'s most dangerous attacker.'
    },
    'James Rodríguez': {
        'team': 'Colombia', 'position': 'CAM', 'age': 34,
        'club': 'Rayo Vallecano', 'impact': 7.5,
        'goals_per_game': 0.2, 'assists_per_game': 0.45,
        'rating': 7.6,
        'note': 'WC 2014 golden boot winner. Still has moments of brilliance. Colombia\'s playmaking brain.'
    },

    # ── UNDERDOG STARS ────────────────────────────────────────────
    'Akira Kubo': {
        'team': 'Japan', 'position': 'RW', 'age': 23,
        'club': 'Real Sociedad', 'impact': 7.8,
        'goals_per_game': 0.3, 'assists_per_game': 0.35,
        'rating': 7.8,
        'note': 'Creative spark for Japan. Technical quality in tight spaces.'
    },
    'Nayef Aguerd': {
        'team': 'Morocco', 'position': 'CB', 'age': 28,
        'club': 'West Ham', 'impact': 8.0,
        'goals_per_game': 0.1, 'assists_per_game': 0.05,
        'rating': 8.1,
        'note': 'Morocco\'s defensive rock alongside Saiss. Premier League quality center back.'
    },
    'Said Benrahma': {
        'team': 'Algeria', 'position': 'LW', 'age': 29,
        'club': 'Lyon', 'impact': 7.6,
        'goals_per_game': 0.28, 'assists_per_game': 0.32,
        'rating': 7.7,
        'note': 'Tricky, creative. Algeria\'s most dynamic attacker alongside Mahrez. Underrated.'
    },
    'Ismaila Sarr': {
        'team': 'Senegal', 'position': 'RW', 'age': 26,
        'club': 'Marseille', 'impact': 8.0,
        'goals_per_game': 0.35, 'assists_per_game': 0.3,
        'rating': 8.0,
        'note': 'Pace and power. Senegal\'s second most dangerous attacker after Mané. Direct and explosive.'
    },
    'Sébastien Haller': {
        'team': 'Ivory Coast', 'position': 'ST', 'age': 30,
        'club': 'Borussia Dortmund', 'impact': 8.2,
        'goals_per_game': 0.55, 'assists_per_game': 0.18,
        'rating': 8.1,
        'note': 'Overcame cancer. Powerful striker. Ivory Coast\'s goal threat and emotional leader.'
    },
    'Franck Kessié': {
        'team': 'Ivory Coast', 'position': 'CM', 'age': 29,
        'club': 'Al Ahli', 'impact': 7.8,
        'goals_per_game': 0.15, 'assists_per_game': 0.2,
        'rating': 7.8,
        'note': 'Physical powerhouse in midfield. Ivory Coast\'s engine room. Former Barcelona player.'
    },
    'Amath Ndiaye': {
        'team': 'Senegal', 'position': 'RW', 'age': 24,
        'club': 'Girona', 'impact': 7.5,
        'goals_per_game': 0.3, 'assists_per_game': 0.28,
        'rating': 7.5,
        'note': 'Rising star. Exciting young talent in Senegal\'s attack.'
    },
}

# ═══════════════════════════════════════════════════════════════════
# TEAM STAR POWER AGGREGATION
# ═══════════════════════════════════════════════════════════════════

def get_team_players(team):
    return {name: p for name, p in PLAYERS.items() if p.get('team') == team}

def team_star_power(team):
    players = get_team_players(team)
    if not players: return 0
    impacts = [p['impact'] for p in players.values()]
    return round(np.mean(impacts) + (max(impacts) - np.mean(impacts)) * 0.4, 2)

def team_uplift(team):
    """How much above average are they due to player quality?"""
    sp = team_star_power(team)
    avg = 7.5
    return round((sp - avg) / avg * 100, 1)

# ═══════════════════════════════════════════════════════════════════
# UNDERDOG ANALYSIS — teams where players > team reputation
# ═══════════════════════════════════════════════════════════════════

UNDERDOG_TEAMS = {
    'Norway':       {'fifa_rank': 24, 'group': 'H', 'base_strength': 5.5},
    'Morocco':      {'fifa_rank': 10, 'group': 'C', 'base_strength': 7.2},
    'Algeria':      {'fifa_rank': 27, 'group': 'I', 'base_strength': 5.8},
    'South Korea':  {'fifa_rank': 22, 'group': 'L', 'base_strength': 6.5},
    'Japan':        {'fifa_rank': 11, 'group': 'E', 'base_strength': 7.0},
    'Ivory Coast':  {'fifa_rank': 39, 'group': 'D', 'base_strength': 6.2},
    'Senegal':      {'fifa_rank': 17, 'group': 'H', 'base_strength': 6.8},
    'Canada':       {'fifa_rank': 25, 'group': 'A', 'base_strength': 6.0},
    'United States':{'fifa_rank': 15, 'group': 'B', 'base_strength': 6.5},
    'Uruguay':      {'fifa_rank': 13, 'group': 'G', 'base_strength': 6.8},
}

# ═══════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("  FIFA WORLD CUP 2026 — PLAYER INTELLIGENCE REPORT")
print("=" * 70)

# ── ALL PLAYERS RANKED ────────────────────────────────────────────────────────
print("\n🌟 TOP PLAYERS AT THE TOURNAMENT (by impact rating)\n")
print(f"  {'Player':<22} {'Team':<18} {'Pos':<5} {'Club':<22} {'Rating':>6} {'Impact':>7}")
print(f"  {'-'*82}")

sorted_players = sorted(
    [(name, p) for name, p in PLAYERS.items() if p.get('team')],
    key=lambda x: x[1]['rating'], reverse=True
)
for name, p in sorted_players:
    print(f"  {name:<22} {p['team']:<18} {p['position']:<5} {p['club']:<22} {p['rating']:>6.1f} {p['impact']:>7.1f}")

# ── UNDERDOG TEAM DEEP DIVES ──────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("  🔥 UNDERDOG TEAM ANALYSIS — Why They Can Surprise Everyone")
print("=" * 70)

underdog_analysis = [
    {
        'team': 'Norway',
        'group': 'H',
        'group_opponents': ['France', 'Senegal', 'Iraq'],
        'danger_level': '🔴 Very High',
        'verdict': 'DARK HORSE TO REACH KNOCKOUTS',
        'analysis': """
    Norway are the most dangerous underdog at WC 2026. Here's why:

    ⚡ HAALAND FACTOR:
       Erling Haaland is arguably the best striker alive. At 25, he is at his
       physical and technical peak. Man City's system has made him unstoppable —
       0.91 goals per game is extraordinary. In a tournament format where one
       moment wins games, Haaland is that moment every single time.

    🎨 ØDEGAARD LINK-UP:
       Martin Ødegaard is a world-class 10. He reads the game brilliantly and
       will find Haaland in dangerous positions. The Arsenal captain-Haaland
       combination could be the partnership of the tournament.

    📊 GROUP H REALITY CHECK:
       - vs Iraq (Jun 16): Norway should WIN. Iraq are weak. Haaland will score.
       - vs France (Jun 26): Massive test. But Norway CAN hold France and
         nick a result. Haaland scored against PSG — he can score against France.
       - vs Senegal (Jun 22): 50/50. This is the key match for qualification.

    🎯 REALISTIC OUTCOME: Norway qualify from Group H in 2nd place.
       In knockouts, with Haaland in form, they can beat ANYONE on their day.
    """,
    },
    {
        'team': 'Morocco',
        'group': 'C',
        'group_opponents': ['Brazil', 'Haiti', 'Scotland'],
        'danger_level': '🔴 Very High',
        'verdict': 'GENUINE SEMIFINAL CONTENDERS',
        'analysis': """
    Morocco proved at Qatar 2022 they are not underdogs anymore — but the world
    still underestimates them. Our model has them beating Brazil. Here's why
    that's actually plausible:

    🛡️ DEFENSIVE FORTRESS:
       Hakimi + Aguerd + Amrabat = the best defensive unit in Africa and one of
       the best globally. Morocco conceded just 1 goal in the 2022 knockouts.
       That defensive structure is still intact and now more experienced.

    ⚡ HAKIMI ON THE RIGHT:
       Achraf Hakimi is the best right back in the world at PSG. He offers a
       constant attacking outlet and shuts down opposition left wingers. World class.

    🇧🇷 WHY THEY CAN BEAT BRAZIL:
       Brazil without a settled system are beatable. Morocco will sit deep,
       hit on the counter with Hakimi bombing forward, and trust Amrabat to
       nullify Brazilian creativity. They did this to SPAIN and PORTUGAL.

    🎯 REALISTIC OUTCOME: GROUP WINNERS. Potential semifinal run.
    """,
    },
    {
        'team': 'Japan',
        'group': 'E',
        'group_opponents': ['Netherlands', 'Sweden', 'Tunisia'],
        'danger_level': '🟠 High',
        'verdict': 'KNOCKOUTS GUARANTEED. QUARTER FINAL POSSIBLE.',
        'analysis': """
    Japan are no longer a surprise. They are a GENUINE footballing nation now.
    2022: Beat Germany 2-1. Beat Spain 2-1. Reached knockouts.

    🌀 MITOMA IS THE KEY:
       Kaoru Mitoma at Brighton is one of Europe's most exciting wingers. His
       dribbling stats are elite — low centre of gravity, unpredictable, two-footed.
       He will cause the Netherlands real problems.

    ⚙️ ORGANISATION IS ELITE:
       Japan press harder than almost any team at the tournament. Their high
       press caused Germany's 2022 collapse. Wataru Endo (Liverpool) screens
       the back four expertly. They are hard to break down AND hard to play against.

    📊 GROUP E BREAKDOWN:
       - vs Tunisia (Jun 20): WIN. Mitoma should run riot.
       - vs Sweden (Jun 25): STRONG WIN. Model says 3-0. Realistic.
       - vs Netherlands (Jun 14): Can pull off a shock. Remember 2022.

    🎯 REALISTIC OUTCOME: GROUP WINNERS. Potential quarter final.
    """,
    },
    {
        'team': 'South Korea',
        'group': 'L',
        'group_opponents': ['Czech Republic', 'South Africa', 'Mexico'],
        'danger_level': '🟠 High',
        'verdict': 'SON CAN CARRY THEM TO KNOCKOUTS',
        'analysis': """
    Son Heung-min at 34 is still elite. This is his last realistic World Cup
    and he will be motivated like never before.

    👑 SON HEUNG-MIN FACTOR:
       Captain. Leader. 0.55 goals per game. He has the pace, technique and big-game
       mentality to win matches single-handedly. Watch him in the big moments.

    🎨 LEE KANG-IN SUPPORT:
       The PSG midfielder is 23 and improving rapidly. His creativity alongside
       Son gives South Korea a dangerous attacking duo.

    📊 GROUP L LOOKS MANAGEABLE:
       - vs South Africa (Jun 24): Model gives away win. But with Son, Korea WIN.
       - vs Mexico (Jun 18): Tough. Mexico have more quality overall.
       - vs Czech Republic (Jun 18): Winnable. Son should be decisive.

    🎯 REALISTIC OUTCOME: 2nd place in Group L. Round of 16 exit possible.
    """,
    },
    {
        'team': 'Canada',
        'group': 'A',
        'group_opponents': ['Bosnia and Herzegovina', 'Switzerland', 'Qatar'],
        'danger_level': '🟡 Medium-High',
        'verdict': 'ALPHONSO DAVIES MAKES THEM DANGEROUS',
        'analysis': """
    Canada's first World Cup since 1986 is now their second in a row. They are
    growing fast. Davies + David is a serious attacking duo.

    ⚡ ALPHONSO DAVIES — THE FLASH:
       Bayern Munich's left back is one of the fastest players on earth. His
       acceleration is terrifying. He doesn't just defend — he's a genuine
       attacking weapon who can score and assist from left back.

    🎯 JONATHAN DAVID — THE FINISHER:
       0.75 goals per game in Ligue 1. Clinical. If Davies sets him up, he scores.
       Simple as that. The question is whether Canada can create those chances.

    📊 GROUP A OUTLOOK:
       - vs Bosnia and Herzegovina: WIN. Our model agrees (71% confident).
       - vs Switzerland: Tough. Switzerland are solid and organized.
       - vs Qatar: WIN. Qatar are weak.

    🎯 REALISTIC OUTCOME: Qualify from Group A in 2nd place.
    """,
    },
    {
        'team': 'Algeria',
        'group': 'I',
        'group_opponents': ['Argentina', 'Austria', 'Jordan'],
        'danger_level': '🟡 Medium',
        'verdict': 'TOUGHEST GROUP BUT MAHREZ MAGIC IS REAL',
        'analysis': """
    Algeria are in the group of death with Argentina. But our model
    PREDICTS ALGERIA TO BEAT ARGENTINA (46%). Here's why that's not crazy:

    🪄 MAHREZ AT 35 — STILL MAGIC:
       Riyad Mahrez in big games can produce moments of genius. Algeria
       have always been a counter-attacking team and Mahrez is perfect
       for that role — hold the ball, beat a man, create a goal.

    🔵 BENRAHMA IS THE WILD CARD:
       Said Benrahma at Lyon is in good form. He adds creativity and
       is unpredictable. Algeria's attack has more depth than people think.

    🇦🇷 HOW TO BEAT ARGENTINA:
       Messi is 39. Lautaro is the real threat. But Algeria can sit deep,
       frustrate them, and hit on the counter. They did it to Germany
       in 2014 (drew 1-1 in 90 mins, lost in extra time). Same plan works.

    🎯 REALISTIC OUTCOME: 3rd place. Could sneak through as best 3rd place team.
    """,
    },
]

for team_data in underdog_analysis:
    team = team_data['team']
    players = get_team_players(team)
    sp = team_star_power(team)
    uplift = team_uplift(team)
    print(f"\n{'─'*70}")
    print(f"  🏴 {team.upper()} — Group {team_data['group']}")
    print(f"  Danger Level: {team_data['danger_level']}   |   Verdict: {team_data['verdict']}")
    print(f"  Star Power Rating: {sp:.1f}/10   |   Player Uplift vs Average: +{uplift}%")
    print(f"\n  Key Players:")
    for name, p in sorted(players.items(), key=lambda x: x[1]['impact'], reverse=True):
        print(f"    ⭐ {name:<22} ({p['position']}, {p['club']}) — Impact: {p['impact']:.1f}/10")
        print(f"       \"{p['note']}\"")
    print(team_data['analysis'])

# ── PLAYER IMPACT ON MATCH PREDICTIONS ───────────────────────────────────────
print("\n" + "=" * 70)
print("  📊 PLAYER IMPACT ADJUSTMENTS — Where Stars Shift the Odds")
print("=" * 70)

adjustments = [
    ("Norway vs France",     "Haaland + Ødegaard",  "Model: France Win 41%", "Adjusted: 45-48% Norway chance. Haaland makes any scoreline possible."),
    ("Norway vs Senegal",    "Haaland vs Mané",     "Model: Draw 47%",       "Both teams have game-changers. Genuinely 50/50. Back the draw at 2.15."),
    ("Japan vs Netherlands", "Mitoma vs Van Dijk",  "Model: Japan Win 39%",  "Mitoma's dribbling could unlock even Van Dijk. Watch for Japan shock."),
    ("Algeria vs Argentina", "Mahrez vs Messi",     "Model: Algeria 46%",    "At 39, Messi is beatable over 90 mins. Algeria's counter + Mahrez = threat."),
    ("Morocco vs Brazil",    "Hakimi vs Vinicius",  "Model: Morocco Win 52%","Morocco's defensive system has beaten bigger teams. This is REAL."),
    ("South Korea vs Mexico","Son vs Mexico midfield","Model: Draw 40%",     "Son's big-game mentality tips this. Back South Korea or BTTS."),
    ("Canada vs Switzerland","Davies vs Swiss defence","Model: Canada Win 46%","Davies is unplayable on his day. Switzerland haven't faced his pace."),
    ("Egypt vs Iran",        "Salah vs Iran defence","Model: Egypt Win 45%", "Salah is too good for Group F. Egypt should top their matches."),
    ("Uruguay vs Spain",     "Valverde vs Pedri",   "Model: Spain Win 55%", "Best midfield battle of the group stage. Valverde can match Pedri."),
    ("Senegal vs Iraq",      "Mané + Sarr",         "Model: Senegal Win 57%","Strong confidence. Senegal have the quality. Back them."),
]

print(f"\n  {'Match':<30} {'Key Duel':<28} {'Model Says':<25} Analysis")
print(f"  {'-'*110}")
for match, duel, model, analysis in adjustments:
    print(f"\n  {match:<30} {duel:<28} {model:<25}")
    print(f"  → {analysis}")

# ── GOLDEN BOOT PREDICTIONS ───────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("  🥇 GOLDEN BOOT PREDICTIONS")
print("=" * 70)
print("""
  Rank  Player              Team          Pred Goals  Reasoning
  ────  ──────────────────  ────────────  ──────────  ─────────────────────────
  1st   Erling Haaland      Norway        6-8 goals   Peak striker, hungry for WC glory
  2nd   Kylian Mbappé        France        5-7 goals   France go deep, he scores in every round
  3rd   Lamine Yamal        Spain         4-6 goals   18yo phenom, Spain's best player
  4th   Harry Kane          England       4-6 goals   Clinical. England should advance far
  5th   Vinicius Jr         Brazil        4-6 goals   Even in poor Brazil form, he creates
  6th   Mohamed Salah       Egypt         4-5 goals   Carries Egypt. Motivated to prove himself
  7th   Lautaro Martínez    Argentina     4-5 goals   Argentina's real striker threat
  8th   Kaoru Mitoma        Japan         3-5 goals   Will be the tournament's breakout star
  9th   Jonathan David      Canada        3-4 goals   Clinical finisher, good group draw
  10th  Luis Díaz           Colombia      3-4 goals   Liverpool form translating to NT
""")

print("=" * 70)
print("  ⚠️  Disclaimer: Player analysis based on 2025-26 season form.")
print("  Injuries, suspensions and team tactics can change everything.")
print("=" * 70)
