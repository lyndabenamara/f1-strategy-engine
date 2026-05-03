import fastf1
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

fastf1.Cache.enable_cache('f1_cache')

DEFAULT_DEG_RATE = 0.05
DEFAULT_PIT_DELTA = 22
MIN_STINT_LENGTH = 10
FINAL_STINT_LAPS = 5
MAX_TYRE_LIFE = 25
DEG_RATE_FILTER = 0.5
RACE_FRACTION_CUTOFF = 0.67

# data loading 
def load_multiple_races(track, years=[2022, 2023, 2024, 2025]):
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

    if not all_laps:
        raise ValueError(f"No data found for track: {track}. Check spelling and try again.")

    return pd.concat(all_laps, ignore_index=True), total_laps

# ── degradation modelling ──────────────────────────────────────────────────────
def clean_laps(laps, compound):
    df = laps[
        (laps['Compound'] == compound) &
        (laps['PitInTime'].isna()) &
        (laps['PitOutTime'].isna()) &
        (laps['LapTime'].notna())
    ].copy()
    df = df[df['TyreLife'] > 1].copy()
    df = df[df['TyreLife'] <= MAX_TYRE_LIFE].copy()

    if df.empty:
        return df

    # only use first 2/3 of race to avoid end-of-race push distortion
    max_lap = df['LapNumber'].max()
    df = df[df['LapNumber'] <= (max_lap * RACE_FRACTION_CUTOFF)].copy()
    df['LapTimeSeconds'] = df['LapTime'].dt.total_seconds()

    # remove outlier lap times (safety cars, incidents etc)
    q_low = df['LapTimeSeconds'].quantile(0.05)
    q_high = df['LapTimeSeconds'].quantile(0.95)
    df = df[(df['LapTimeSeconds'] >= q_low) & (df['LapTimeSeconds'] <= q_high)].copy()

    return df

def get_deg_rate_lr(laps, compound):
    df = clean_laps(laps, compound)

    if df.empty:
        return None

    stint_rates = []
    for (year, driver, stint), stint_df in df.groupby(['Year', 'Driver', 'Stint']):
        if len(stint_df) < 4:
            continue
        X = stint_df['TyreLife'].values.reshape(-1, 1)
        y = stint_df['LapTimeSeconds'].values
        model = LinearRegression()
        model.fit(X, y)
        rate = model.coef_[0]
        if -DEG_RATE_FILTER < rate < DEG_RATE_FILTER:
            stint_rates.append(rate)

    if len(stint_rates) < 3:
        return None

    return round(float(np.median(stint_rates)), 3)

# ── pit delta ──────────────────────────────────────────────────────────────────
def calculate_pit_delta(laps):
    pit_laps = laps[laps['PitInTime'].notna() & laps['PitOutTime'].notna()].copy()
    if pit_laps.empty:
        return DEFAULT_PIT_DELTA
    pit_laps['PitDelta'] = (pit_laps['PitOutTime'] - pit_laps['PitInTime']).dt.total_seconds()
    pit_laps = pit_laps[(pit_laps['PitDelta'] > 15) & (pit_laps['PitDelta'] < 35)]
    if pit_laps.empty:
        return DEFAULT_PIT_DELTA
    return round(float(pit_laps['PitDelta'].median()), 1)

# ── strategy logic ─────────────────────────────────────────────────────────────
def calculate_stint_cost(compound, num_laps, deg_rates, current_tyre_age=0):
    rate = deg_rates.get(compound, DEFAULT_DEG_RATE)
    if rate is None:
        rate = DEFAULT_DEG_RATE
    total_cost = 0
    for lap in range(num_laps):
        total_cost += rate * (current_tyre_age + lap)
    return round(total_cost, 2)

def check_pit_window(gap_behind, gap_ahead, pit_stop_delta):
    safe_to_pit = gap_behind > pit_stop_delta
    undercut_possible = 0 < gap_ahead < 3.0
    return safe_to_pit, undercut_possible

def get_conversational_recommendation(current_lap, total_laps, current_compound,
                                       tyre_age, position, gap_ahead, gap_behind,
                                       recommendations, safe_to_pit, deg_rates, pit_stop_delta):
    best = recommendations[0]
    laps_remaining = total_laps - current_lap
    rate = deg_rates.get(current_compound) or DEFAULT_DEG_RATE

    if best['option'] == 'STAY OUT':
        if rate < 0:
            msg = f"Stay out. Your {current_compound.lower()}s are actually getting faster as the track rubbers in — no reason to pit over {laps_remaining} laps."
        else:
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
        deg_advantage = round(abs(rate) * tyre_age, 2)
        laps_to_catch = round(time_behind / deg_advantage) if deg_advantage > 0 else '?'
        msg = f"Pit for {compound}s despite the tight gap — you'll rejoin {time_behind}s behind but gain ~{deg_advantage}s/lap on fresher rubber, catching back up in ~{laps_to_catch} laps with {laps_remaining} still to go."

    if gap_ahead > 0 and gap_ahead < 3.0:
        msg += f" With {gap_ahead}s to the car ahead, fresh tyres could set up an undercut."

    return msg

def recommend_strategy(current_lap, total_laps, current_compound,
                       tyre_age, position, gap_ahead, gap_behind, deg_rates, pit_stop_delta):

    laps_remaining = total_laps - current_lap

    # edge case: invalid lap number
    if current_lap >= total_laps:
        print(f"\n🏁 Race is over!")
        return

    # edge case: too early to pit
    if tyre_age < MIN_STINT_LENGTH:
        print(f"\n--- Strategy Report: Lap {current_lap}/{total_laps} ---")
        print(f"Current: P{position} on {current_compound} (age: {tyre_age} laps)")
        print(f"\n🟢 Too early to pit — tyres only {tyre_age} laps old, stay out and build the gap.")
        return

    recommendations = []

    # option 1: stay out
    stay_out_cost = calculate_stint_cost(current_compound, laps_remaining, deg_rates, tyre_age)
    stay_out_label = f'Tyre benefit to end: {abs(stay_out_cost):.1f}s' if stay_out_cost < 0 else f'Deg cost to end: {stay_out_cost:.1f}s'
    recommendations.append({
        'option': 'STAY OUT',
        'compound': current_compound,
        'cost': stay_out_cost,
        'reason': stay_out_label
    })

    # option 2: pit for each other compound
    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        if compound == current_compound:
            continue
        # edge case: not enough laps left to justify pitting for this compound
        pit_cost = pit_stop_delta + calculate_stint_cost(compound, laps_remaining, deg_rates, 0)
        pit_label = f'Pit delta - tyre benefit: {abs(pit_cost):.1f}s' if pit_cost < 0 else f'Pit delta + deg cost: {pit_cost:.1f}s'
        recommendations.append({
            'option': f'PIT FOR {compound}',
            'compound': compound,
            'cost': pit_cost,
            'reason': pit_label
        })

    recommendations.sort(key=lambda x: x['cost'])

    # edge case: final stint
    if laps_remaining <= FINAL_STINT_LAPS and recommendations[0]['option'] == 'STAY OUT':
        print(f"\n--- Strategy Report: Lap {current_lap}/{total_laps} ---")
        print(f"Current: P{position} on {current_compound} (age: {tyre_age} laps)")
        print(f"\n🏁 Final stint — {laps_remaining} laps to go, stay out and bring it home.")
        return

    safe_to_pit, undercut_possible = check_pit_window(gap_behind, gap_ahead, pit_stop_delta)

    # print report
    print(f"\n--- Strategy Report: Lap {current_lap}/{total_laps} ---")
    print(f"Current: P{position} on {current_compound} (age: {tyre_age} laps)")

    if gap_ahead == 0:
        print(f"Running in the lead | Gap behind: {gap_behind}s")
    elif gap_behind == 0:
        print(f"Gap ahead: {gap_ahead}s | Running last")
    else:
        print(f"Gap ahead: {gap_ahead}s | Gap behind: {gap_behind}s")

    print(f"Pit delta at this track: {pit_stop_delta}s")
    print(f"\nOptions (best to worst):")
    for i, rec in enumerate(recommendations):
        marker = " ← RECOMMENDED" if i == 0 else ""
        print(f"  {rec['option']}: {rec['reason']}{marker}")

    # undercut warning
    if undercut_possible:
        if recommendations[0]['option'] != 'STAY OUT':
            print(f"\n⚠️  Undercut opportunity! {gap_ahead}s to car ahead — pit now and you may come out ahead on fresh tyres")
        else:
            print(f"\n💡 Undercut possible ({gap_ahead}s to car ahead) but staying out is cheaper — monitor the gap")

    # pit window safety
    if not safe_to_pit and recommendations[0]['option'] != 'STAY OUT':
        best_pit = next((r for r in recommendations if r['option'] != 'STAY OUT'), None)
        if best_pit:
            time_behind_after_pit = pit_stop_delta - gap_behind
            rate = deg_rates.get(current_compound) or DEFAULT_DEG_RATE
            deg_advantage_per_lap = abs(rate) * tyre_age
            laps_to_catch = round(time_behind_after_pit / deg_advantage_per_lap) if deg_advantage_per_lap > 0 else '?'
            print(f"\n⚠️  Risky pit! You'd rejoin ~{time_behind_after_pit:.1f}s behind the car behind.")
            print(f"   On fresh tyres you'd gain ~{deg_advantage_per_lap:.2f}s/lap — catching back up in ~{laps_to_catch} laps")
    elif safe_to_pit and recommendations[0]['option'] != 'STAY OUT':
        print(f"\n✅  Safe pit window — {gap_behind}s gap behind is enough to cover the stop")

    # edge case: leading — no gap ahead
    if gap_ahead == 0 and recommendations[0]['option'] != 'STAY OUT':
        print(f"\n🏆 You're leading — pit on your own terms, no undercut threat")

    print(f"\n🏎️  Strategist says:")
    print(get_conversational_recommendation(current_lap, total_laps, current_compound,
                                             tyre_age, position, gap_ahead, gap_behind,
                                             recommendations, safe_to_pit, deg_rates, pit_stop_delta))

# ── user inputs ────────────────────────────────────────────────────────────────
def get_user_inputs():
    print("\n🏎️  F1 Strategy Engine")
    print("=" * 30)

    track = input("Track name (e.g. Bahrain, Monza, Silverstone): ").strip()

    while True:
        try:
            current_lap = int(input("Current lap: "))
            if current_lap < 1:
                print("Lap must be at least 1.")
                continue
            break
        except ValueError:
            print("Please enter a number.")

    while True:
        current_compound = input("Current compound (SOFT/MEDIUM/HARD): ").strip().upper()
        if current_compound in ['SOFT', 'MEDIUM', 'HARD']:
            break
        print("Please enter SOFT, MEDIUM, or HARD.")

    while True:
        try:
            tyre_age = int(input("Tyre age (laps): "))
            if tyre_age < 1:
                print("Tyre age must be at least 1.")
                continue
            break
        except ValueError:
            print("Please enter a number.")

    while True:
        try:
            position = int(input("Current position (1-22): "))
            if not 1 <= position <= 22:
                print("Position must be between 1 and 22.")
                continue
            break
        except ValueError:
            print("Please enter a number.")

    gap_ahead = 0.0
    if position > 1:
        while True:
            try:
                gap_ahead = float(input("Gap to car ahead (seconds): "))
                if gap_ahead < 0:
                    print("Gap must be positive.")
                    continue
                break
            except ValueError:
                print("Please enter a number.")

    gap_behind = 0.0
    if position < 20:
        while True:
            try:
                gap_behind = float(input("Gap to car behind (seconds): "))
                if gap_behind < 0:
                    print("Gap must be positive.")
                    continue
                break
            except ValueError:
                print("Please enter a number.")

    return track, current_lap, current_compound, tyre_age, position, gap_ahead, gap_behind

# ── main ───────────────────────────────────────────────────────────────────────
track, current_lap, current_compound, tyre_age, position, gap_ahead, gap_behind = get_user_inputs()

try:
    laps, total_laps = load_multiple_races(track)
except ValueError as e:
    print(f"\n❌ {e}")
    exit()

if current_lap > total_laps: # type: ignore
    print(f"\n❌ Lap {current_lap} is beyond the race distance of {total_laps} laps.")
    exit()

deg_rates = {
    'SOFT': get_deg_rate_lr(laps, 'SOFT'),
    'MEDIUM': get_deg_rate_lr(laps, 'MEDIUM'),
    'HARD': get_deg_rate_lr(laps, 'HARD')
}

for compound in ['SOFT', 'MEDIUM', 'HARD']:
    if deg_rates[compound] is None:
        print(f"⚠️  No valid {compound} data for this track, using default deg rate ({DEFAULT_DEG_RATE}s/lap)")
        deg_rates[compound] = DEFAULT_DEG_RATE

pit_stop_delta = calculate_pit_delta(laps)

print(f"\n📍 {track} Grand Prix — {total_laps} laps total")
print(f"Deg rates: SOFT={deg_rates['SOFT']}s/lap | MEDIUM={deg_rates['MEDIUM']}s/lap | HARD={deg_rates['HARD']}s/lap")
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