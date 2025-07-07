# Batsman Score Predictor
Builds a predictive model to forecast a cricketer's (bastman) number of runs and  whether they will score above/below a run target, using historical data. 

### Project Overview

The project forecasts how many runs a batsman (such as Virat Kohli or Harry Brook) is likely to score in an upcoming match and whether they will score above/below a target. It uses:

- Ball-by-ball and match-level cricket data (Cricsheet JSON)
- Feature engineering (e.g. batting average vs opposition, recent form, venue)
- Regression and classification models to:
  - Predict **expected runs scored**
  - Decide whether the player will **exceed a target threshold** (e.g., 30 runs)


### How to Use
Download the data folder, getdata.py and predict.py
In getdata.py change player_name to the player of your choice (note: player names have to be inputed in a specific way, eg. V Kohli, HC Brook, JC Buttler - future iterations of this project will automate this process)
Then in predict.py change player_name and run_target to those of your choice. Also, change new_match_data to reflect a new match


### Thoughts
The regression ML model to predict the number of runs scored is not very accurate - possible causes:
- Lack of data (even data from 100 matches may not be enough)
- Lack of features - we can implement more features in future iterations, however, there are lots of features (such as ball speed) I cannot get data on
- Unpredictability of sport - especially given this is for T20 matches, the most unpredictable out of all cricket formats
  


