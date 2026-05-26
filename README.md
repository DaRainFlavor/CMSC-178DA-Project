# Philippine Labor Market Polarization Dashboard (2016 – 2025)

An interactive, high-performance Streamlit data visualization dashboard built by Team **Inverters** for the **CMSC 178DA Capstone Project**. This application analyzes whether the Philippine labor market has polarized into an "hourglass" structure, analyzing employment share changes, skill tier trends, and advanced projections through 2030.

---

## 🚀 Key Features

* **Overview & Finding Calculator**: Computes structural changes across High, Middle, and Low Skill tiers with detailed PSA-adjusted data models.
* **Skill Tier Trends**: Visualization of absolute employment dynamics and percentage share shifts over time.
* **Trend Projections**: Forecasts 2030 outlook using linear or quadratic regression fits alongside YoY share changes.
* **Occupation Clustering**: Utilizes **K-Means Clustering** to segment occupations by CAGR, volatility, and average employment metrics.
* **Premium Design System**: Fully customized stylesheet with sleek glassmorphism panels, interactive badges, micro-animations, and synchronized double-column flex layouts.

---

## 🛠️ Installation & Setup

Follow these steps to clone the repository, set up a virtual environment, and launch the application:

### 1. Clone the Repository
```bash
git clone https://github.com/savvyavie/CMSC-178DA-Project.git
cd CMSC-178DA-Project
```

### 2. Set Up a Virtual Environment (Recommended)
Creating an isolated virtual environment prevents library version conflicts:

* **On Windows**:
  ```powershell
  python -m venv venv
  venv\Scripts\activate
  ```
* **On macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
Install all required libraries specified in the `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 💻 Running the Dashboard

Once the virtual environment is activated and dependencies are installed, you can start the Streamlit server:

* **Via Streamlit command**:
  ```bash
  streamlit run app.py
  ```
* **Via Windows Batch Script**:
  Simply double-click the `run_dashboard.bat` file in the project folder to automatically activate the environment and launch the dashboard.

Open the URL shown in your terminal (usually `http://localhost:8501`) in your browser to inspect the application.

---

## 📊 Technologies Used

* **Core**: [Streamlit](https://streamlit.io/) (Python Framework)
* **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
* **Visualization**: [Plotly Go / Express](https://plotly.com/)
* **Analytics**: [Scikit-learn](https://scikit-learn.org/) (K-Means Clustering, StandardScaler)
* **Design & Styling**: Custom CSS layout injection with flexbox alignment
