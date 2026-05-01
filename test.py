import fastf1
session = fastf1.get_session(2026, 'Japan', 'R')
session.load()
laps = session.laps
print(drivers := laps['Driver'].unique())