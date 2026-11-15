# 🏏 IPL Live Win Probability Predictor

An end-to-end Machine Learning web application that predicts real-time win probabilities for chasing teams in Indian Premier League (IPL) matches based on live second-innings match situations.

## 🚀 Live Web App
👉 **[Click here to test the Live Predictor](https://ipl-win-predictor-sneha.streamlit.app/)**

## 📊 Project Highlights
- **Dataset:** IPL Complete Dataset (2008–2024) with 260,000+ ball-by-ball delivery records.
- **Feature Engineering:** Created live match features including `runs_left`, `balls_left`, `wickets_left`, `current_run_rate (CRR)`, and `required_run_rate (RRR)`.
- **Model Performance:** Built a preprocessing and classification pipeline (`OneHotEncoder`, `StandardScaler`, and `LogisticRegression`) achieving **~80% prediction accuracy**.
- **Deployment:** Developed an interactive interface with **Streamlit** and deployed it on **Streamlit Community Cloud**.

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, Pickle
- **Deployment:** Streamlit
