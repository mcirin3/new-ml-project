# Fantasy Football ML Lineup Optimizer — Implementation Documentation

## Overview
This project is a **Streamlit web application** that optimizes a fantasy football lineup using **ESPN public projections** or a **Custom ML model**, merges historical 2024 season totals with Year-To-Date (YTD) data, and displays head-to-head matchups with the selected week’s opponent.

---

## Features Implemented 

### 1. Data Loading & Preprocessing
- **2024 Season Totals**:
  - Loaded from `data/raw/espn/espn_players_2024_scoring4.csv`.
  - Normalized and merged with current season data using a name normalization function.
- **YTD Player Totals**:
  - Pulled from ESPN public APIs (`view=kona_player_info`), filtered for current week.
- **Name Normalization**:
  - Lowercased names, removed punctuation, suffixes (`Jr`, `Sr`), and hyphens for reliable merges.

### 2. Projection Sources
- **ESPN Public Projections** (default).
- **Custom ML Model**:
  - Can be trained on historical player stats.
  - Filters predictions to only include your current roster.

### 3. Roster Optimization
- Built using **PuLP** for linear programming.
- Slots & constraints configured:
  - QB: 1  
  - RB: 2  
  - WR: 2  
  - TE: 1  
  - FLEX: 1 (RB/WR/TE)  
- Objective function:
  - Maximize expected points, minus a small uncertainty penalty.

### 4. Opponent Identification & Head-to-Head Display
- Fetches opponent for the selected week using ESPN API.
- Retrieves opponent roster and optimizes their lineup using the same constraints.
- Displays:
  - Opponent team name.
  - Opponent starters and bench.
  - Expected starter totals for both teams.

### 5. UI Features
- **Projection Source Selector** (`ESPN Public` or `Custom ML`).
- **Week Selector** (1–18).
- **Your Players Table** with 2024 totals and YTD stats.
- **Optimized Starters & Bench** for your team.
- **Opponent Matchup** table for head-to-head comparisons.
- **Why These Starters?** expander explaining choices.
- Warnings when 2024 totals are missing.

---

## Planned Enhancements 
- **Historical Matchup Results**: Show past matchups and scores.
- **Matchup-Aware Optimization**: Adjust starter choices based on opponent’s strength at each position.
- **Injury/News Integration**: Reduce projections for injured players.
- **Mobile-Optimized UI**.

---

## How to Run the Application 🛠️

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/fantasy-ml-project.git
cd fantasy-ml-project

```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3️⃣ Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ Create and Configure .env file
# Create a `.env` file in the root directory with the following content:

```bash
# .env
ESPN_SWID={YOUR_SWID}
ESPN_S2=YOUR_ESPN_S2_COOKIE
LEAGUE_ID=YOUR_LEAGUE_ID
SEASON=2025
TEAM_ID=YOUR_TEAM_ID
```

### How to find .env file values: 
Step 1 — Get SWID and espn_s2 Cookies
Log into your ESPN Fantasy Football account in Chrome, Edge, or Safari.

Navigate to your fantasy league homepage.

Open Developer Tools:

Mac: View → Developer → Developer Tools (Chrome) or Develop → Show Web Inspector (Safari).

Windows: Press F12.

Go to the Application (or Storage) tab → Cookies section.

Locate and copy:

SWID → Paste into ESPN_SWID (keep {} brackets).

espn_s2 → Paste into ESPN_S2.

## Step 2 — Get LEAGUE_ID
While on your league page, check the URL:

arduino
Copy code
https://fantasy.espn.com/football/team?leagueId=1256311277&teamId=9
The number after leagueId= is your LEAGUE_ID.

### Train Custom Model 
If you want to train a custom model, run the following command:
```bash
cd tools 
python train_custom_model.py
```
### 5️⃣ Run the Application
```bash
streamlit run app.py
```

### 6️⃣ Access the App
Open your web browser and go to:
```
http://localhost:8501
```
## 7️⃣ Stop the Application
To stop the application, press `Ctrl+C` in the terminal where Streamlit is running.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.