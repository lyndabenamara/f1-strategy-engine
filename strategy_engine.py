import fastf1
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

fastf1.Cache.enable_cache('f1_cache')

def calculate_pit_delta(laps):
    pit_laps = laps[laps['PitInTime'].notna() & laps['PitOutTime'].notna()].copy()
    if len(pit_laps) == 0:
        return 22
    pit_laps['PitDelta'] = (pit_laps['PitOutTime'] - pit_laps['PitInTime']).dt.total_seconds()
    pit_laps = pit_laps[(pit_laps['PitDelta'] > 15) & (pit_laps['PitDelta'] < 35)]
    return round(pit_laps['PitDelta'].median(), 1)

def get_conversational_recommendation(current_lap, total_laps, current_compound,
                                       tyre_age, position, gap_ahead, gap_behind,
                                       recommendations, safe_to_pit, deg_rates, pit_stop_delta):
    best = recommendations[0]
    laps_remaining = total_laps - current_lap

    if best['option'] == 'STAY OUT':
        msg = f"Stay out. Your {current_compound.lower()}s have {tyre_age} laps on them but the deg cost of pitting outweighs staying out over {laps_remaining} laps."
        if safe_to_pit:
            msg += f" You have a comfortable {gap_behind}s gap behind so there's no pressure."
        return msg

    compound = best['option'].replace('PIT FOR ', '').lower()
    cost_saving = round(recommendations[1]['cost'] - best['cost'], 1)

    if safe_to_pit:
        msg = f"Box this lap for {compound}s. You have {gap_behind}s behind so you'll come out clean, and it saves you {cost_saving}s over the alternative."
    else:
        time_behind = round(pit_stop_delta - gap_behind, 1)
        deg_advantage = round(deg_rates[current_compound] * tyre_age, 2)
        laps_to_catch = round(time_behind / deg_advantage) if deg_advantage > 0 else '?'
        msg = f"Pit for {compound}s despite the tight gap — you'll rejoin {time_behind}s behind but gain {deg_advantage}s/lap on fresher rubber, catching back up in ~{laps_to_catch} laps with {laps_remaining} still to go."

    if gap_ahead < 3.0:
        msg += f" With {gap_ahead}s to the car ahead, fresh tyres could set up an undercut."

    return msg

def load_multiple_races(track, years=[2022, 2023, 2024]):
    all_laps = []
    total_laps = None
    for year in years:
        try:
            session = fastf1.get_session(year, track, 'R')
            session.load()
            laps = session.laps.copy()
            laps['Year'] = year
            all_laps.append(laps)
            if total_laps is None:
                total_laps = int(laps['LapNumber'].max())
            print(f"Loaded {year} {track}")
        except Exception as e:
            print(f"Could not load {year} {track}: {e}")
    return pd.concat(all_laps, ignore_index=True), total_laps

def clean_laps(laps, compound, max_tyre_life=25):
    df = laps[
        (laps['Compound'] == compound) &
        (laps['PitInTime'].isna()) &
        (laps['PitOutTime'].isna()) &
        (laps['LapTime'].notna())
    ].copy()
    df = df[df['TyreLife'] > 1]
    df = df[df['TyreLife'] <= max_tyre_life]
    df['LapTimeSeconds'] = df['LapTime'].dt.total_seconds()
    return df

def get_deg_rate_lr(laps, compound):
    df = clean_laps(laps, compound)
    stint_rates = []
    for (year, driver, stint), stint_df in df.groupby(['Year', 'Driver', 'Stint']):
        if len(stint_df) < 4:
            continue
        X = stint_df['TyreLife'].values.reshape(-1, 1)
        y = stint_df['LapTimeSeconds'].values
        model = LinearRegression()
        model.fit(X, y)
        rate = model.coef_[0]
        if -0.5 < rate < 0.5:
            stint_rates.append(rate)
    if not stint_rates:
        return None
    return round(np.median(stint_rates), 3)

def calculate_stint_cost(compound, num_laps, deg_rates, pit_stop_delta, current_tyre_age=0):
    rate = deg_rates.get(compound, 0)
    if not rate:
        return 0
    total_cost = 0
    for lap in range(num_laps):
        total_cost += rate * (current_tyre_age + lap)
    return round(total_cost, 2)

def check_pit_window(gap_behind, gap_ahead, pit_stop_delta):
    safe_to_pit = gap_behind > pit_stop_delta
    undercut_possible = gap_ahead < 3.0
    return safe_to_pit, undercut_possible

def recommend_strategy(current_lap, total_laps, current_compound,
                       tyre_age, position, gap_ahead, gap_behind, deg_rates, pit_stop_delta):

    laps_remaining = total_laps - current_lap
    MIN_STINT_LENGTH = 10

    if tyre_age < MIN_STINT_LENGTH:
        print(f"\n--- Strategy Report: Lap {current_lap}/{total_laps} ---")
        print(f"Current: P{position} on {current_compound} (age: {tyre_age} laps)")
        print(f"\n🟢 Too early to pit — tyres only {tyre_age} laps old, stay out and build the gap.")
        return

    recommendations = []

    stay_out_cost = calculate_stint_cost(current_compound, laps_remaining, deg_rates, pit_stop_delta, tyre_age)
    recommendations.append({
        'option': 'STAY OUT',
        'compound': current_compound,
        'cost': stay_out_cost,
        'reason': f'Deg cost to end: {stay_out_cost:.1f}s'
    })

    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        if compound == current_compound:
            continue
        pit_cost = pit_stop_delta + calculate_stint_cost(compound, laps_remaining, deg_rates, pit_stop_delta, 0)
        recommendations.append({
            'option': f'PIT FOR {compound}',
            'compound': compound,
            'cost': pit_cost,
            'reason': f'Pit delta + deg cost: {pit_cost:.1f}s'
        })

    recommendations.sort(key=lambda x: x['cost'])

    if laps_remaining <= 5 and recommendations[0]['option'] == 'STAY OUT':
        print(f"\n--- Strategy Report: Lap {current_lap}/{total_laps} ---")
        print(f"Current: P{position} on {current_compound} (age: {tyre_age} laps)")
        print(f"\n🏁 Final stint — {laps_remaining} laps to go, stay out and bring it home.")
        return

    safe_to_pit, undercut_possible = check_pit_window(gap_behind, gap_ahead, pit_stop_delta)

    print(f"\n--- Strategy Report: Lap {current_lap}/{total_laps} ---")
    print(f"Current: P{position} on {current_compound} (age: {tyre_age} laps)")
    print(f"Gap ahead: {gap_ahead}s | Gap behind: {gap_behind}s")
    print(f"Pit delta at this track: {pit_stop_delta}s")
    print(f"\nOptions (best to worst):")
    for i, rec in enumerate(recommendations):
        marker = " ← RECOMMENDED" if i == 0 else ""
        print(f"  {rec['option']}: {rec['reason']}{marker}")

    if undercut_possible and recommendations[0]['option'] != 'STAY OUT':
        print(f"\n⚠️  Undercut opportunity! {gap_ahead}s to car ahead — pit now and you may come out ahead on fresh tyres")

    if not safe_to_pit and recommendations[0]['option'] != 'STAY OUT':
        best_pit = next((r for r in recommendations if r['option'] != 'STAY OUT'), None)
        if best_pit:
            time_behind_after_pit = pit_stop_delta - gap_behind
            deg_advantage_per_lap = deg_rates[current_compound] * tyre_age
            laps_to_catch = round(time_behind_after_pit / deg_advantage_per_lap) if deg_advantage_per_lap > 0 else '?'
            print(f"\n⚠️  Risky pit! You'd rejoin ~{time_behind_after_pit:.1f}s behind the car behind.")
            print(f"   On fresh tyres you'd gain ~{deg_advantage_per_lap:.2f}s/lap — catching back up in ~{laps_to_catch} laps")
    elif safe_to_pit:
        print(f"\n✅  Safe pit window — {gap_behind}s gap behind is enough to cover the stop")

    print(f"\n🏎️  Strategist says:")
    print(get_conversational_recommendation(current_lap, total_laps, current_compound,
                                             tyre_age, position, gap_ahead, gap_behind,
                                             recommendations, safe_to_pit, deg_rates, pit_stop_delta))

def get_user_inputs():
    print("\n🏎️  F1 Strategy Engine")
    print("=" * 30)
    track = input("Track name (e.g. Bahrain, Monza, Silverstone): ").strip()
    current_lap = int(input("Current lap: "))
    current_compound = input("Current compound (SOFT/MEDIUM/HARD): ").strip().upper()
    tyre_age = int(input("Tyre age (laps): "))
    position = int(input("Current position: "))
    gap_ahead = float(input("Gap to car ahead (seconds, 0 if leading): "))
    gap_behind = float(input("Gap to car behind (seconds, 0 if last): "))
    return track, current_lap, current_compound, tyre_age, position, gap_ahead, gap_behind

# --- main ---
track, current_lap, current_compound, tyre_age, position, gap_ahead, gap_behind = get_user_inputs()

laps, total_laps = load_multiple_races(track)

deg_rates = {
    'SOFT': get_deg_rate_lr(laps, 'SOFT'),
    'MEDIUM': get_deg_rate_lr(laps, 'MEDIUM'),
    'HARD': get_deg_rate_lr(laps, 'HARD')
}

pit_stop_delta = calculate_pit_delta(laps)

print(f"\n📍 {track} Grand Prix — {total_laps} laps total")
print(f"Deg rates: {deg_rates}")
print(f"Pit delta: {pit_stop_delta}s")

recommend_strategy(
    current_lap=current_lap,
    total_laps=total_laps,
    current_compound=current_compound,
    tyre_age=tyre_age,
    position=position,
    gap_ahead=gap_ahead,
    gap_behind=gap_behind,
    deg_rates=deg_rates,
    pit_stop_delta=pit_stop_delta
)