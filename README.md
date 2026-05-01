# F1 Strategy Engine

A mid-race strategy simulator that uses historical Formula 1 tyre 
degradation data to recommend optimal pit stop decisions.

## How it works
- Pulls 3 years of race data from FastF1 for any track
- Models tyre degradation per compound using linear regression
- Given a race snapshot (lap, position, tyre age, gaps), recommends 
  whether to stay out or pit and for which compound
- Detects undercut opportunities and calculates recovery laps

## Usage
python strategy_engine.py

## Stack
Python, FastF1, pandas, scikit-learn