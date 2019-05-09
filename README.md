# TicketML_Project
Final Project for ECE 6254 @ Georgia Tech

The scripts are meant to be run in the following order:
1. feature_extraction_visualization.py
2. Nearest_Gas.py
3. build_neg_data.py
4. ticket_classification.py
5. Poster_plot_gen.py

If you run into trouble executing a script, ensure that the libraries in the requirements.txt file are installed in your python environment. 
If they are not, simply run pip install -r requirements.txt

Traffic_Violations.csv is the original dataset
Traffic_Violations_Features.csv is the result of running script #1
Traffic_Violations_With_5_Times_Negatives.csv is the result of running scripts #2 & #3
