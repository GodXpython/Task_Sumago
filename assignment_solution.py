"""
Assignment: Libraries, ML, SQL
Covers: NumPy, Pandas, Matplotlib & Seaborn, EDA, Machine Learning, SQL
Dataset: employees.csv (included in this repo)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # so plots save to file even without a display
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

PLOTS_DIR = "plots"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Question 1: NumPy (8 Marks)
# ---------------------------------------------------------------------------
def question1_numpy():
    section("QUESTION 1: NumPy")

    arr = np.arange(1, 26).reshape(5, 5)
    print("Array:\n", arr)

    # 1. Shape and dimensions
    print("\nShape:", arr.shape)
    print("Dimensions:", arr.ndim)

    # 2. Second row and fourth column
    print("\nSecond row:", arr[1, :])
    print("Fourth column:", arr[:, 3])

    # 3. Max, min, mean, std
    print("\nMax:", arr.max())
    print("Min:", arr.min())
    print("Mean:", arr.mean())
    print("Std Dev:", arr.std())

    # 4. Replace all even numbers with 0
    modified = arr.copy()
    modified[modified % 2 == 0] = 0

    # 5. Display modified array
    print("\nModified array (evens -> 0):\n", modified)

    return arr, modified


# ---------------------------------------------------------------------------
# Question 2: Pandas (10 Marks)
# ---------------------------------------------------------------------------
def question2_pandas(df):
    section("QUESTION 2: Pandas")

    # 1. First 5 and last 5 records
    print("First 5 records:\n", df.head())
    print("\nLast 5 records:\n", df.tail())

    # 2. Shape and data types
    print("\nShape:", df.shape)
    print("\nData types:\n", df.dtypes)

    # 3. Statistical summary
    print("\nStatistical summary:\n", df.describe())

    # 4. Missing values and duplicates
    print("\nMissing values per column:\n", df.isnull().sum())
    print("\nNumber of duplicate rows:", df.duplicated().sum())

    # 5. New calculated column: Salary per year of experience
    df["SalaryPerExpYear"] = df["Salary"] / df["YearsExperience"].replace(0, np.nan)

    # 6. Filter records: Salary > 60000
    filtered = df[df["Salary"] > 60000]
    print("\nRecords with Salary > 60000:\n", filtered.head())

    # 7. Sort descending by Salary
    sorted_df = df.sort_values("Salary", ascending=False)
    print("\nSorted by Salary (desc), top 5:\n", sorted_df.head())

    # 8. Save modified DataFrame as CSV
    df.to_csv("employees_modified.csv", index=False)
    print("\nSaved modified dataframe to employees_modified.csv")

    return df


# ---------------------------------------------------------------------------
# Question 3: Matplotlib & Seaborn (8 Marks)
# ---------------------------------------------------------------------------
def question3_visualizations(df):
    section("QUESTION 3: Matplotlib & Seaborn")
    clean = df.dropna(subset=["Salary", "Age", "PerformanceScore"])

    # Histogram
    plt.figure(figsize=(6, 4))
    plt.hist(clean["Salary"], bins=10, color="steelblue", edgecolor="black")
    plt.title("Histogram of Salary")
    plt.xlabel("Salary")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/histogram_salary.png")
    plt.close()

    # Bar Chart (average salary by department)
    plt.figure(figsize=(6, 4))
    dept_avg = clean.groupby("Department")["Salary"].mean().sort_values()
    dept_avg.plot(kind="bar", color="coral")
    plt.title("Average Salary by Department")
    plt.ylabel("Average Salary")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/bar_avg_salary_department.png")
    plt.close()

    # Scatter Plot
    plt.figure(figsize=(6, 4))
    plt.scatter(clean["YearsExperience"], clean["Salary"], color="green")
    plt.title("Years of Experience vs Salary")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/scatter_experience_salary.png")
    plt.close()

    # Box Plot
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Department", y="Salary", data=clean)
    plt.title("Salary Distribution by Department")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/boxplot_salary_department.png")
    plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(6, 5))
    numeric_cols = clean.select_dtypes(include=np.number)
    sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/correlation_heatmap.png")
    plt.close()

    print(f"All 5 plots saved to the '{PLOTS_DIR}/' folder.")
    print("""
Observations:
1. Average salary varies noticeably across departments, with some
   departments (e.g. IT/Finance in this sample) trending higher than others.
2. Years of experience and salary show a mild positive relationship,
   but the correlation heatmap indicates it is not very strong in this
   dataset - salary is also driven by department and performance score.
""")


# ---------------------------------------------------------------------------
# Question 4: Exploratory Data Analysis (8 Marks)
# ---------------------------------------------------------------------------
def question4_eda(df):
    section("QUESTION 4: Exploratory Data Analysis (EDA)")

    # 1. Dataset information
    print("Dataset info:")
    df.info()

    # 2. Missing value analysis
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_report = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    print("\nMissing value analysis:\n", missing_report[missing_report["Missing Count"] > 0])

    # 3. Duplicate value analysis
    dup_count = df.duplicated().sum()
    print(f"\nDuplicate rows found: {dup_count}")
    if dup_count > 0:
        print("Duplicate rows:\n", df[df.duplicated()])


# ---------------------------------------------------------------------------
# Question 5: Machine Learning (10 Marks)
# ---------------------------------------------------------------------------
def question5_machine_learning(df):
    section("QUESTION 5: Machine Learning")

    ml_df = df.dropna(subset=["Salary", "Age", "YearsExperience", "PerformanceScore"]).copy()
    ml_df = pd.get_dummies(ml_df, columns=["Department"], drop_first=True)

    feature_cols = [c for c in ml_df.columns
                     if c not in ("EmployeeID", "Name", "Salary", "SalaryPerExpYear")]

    # 1. Independent (X) and dependent (y) variables
    X = ml_df[feature_cols]
    y = ml_df["Salary"]

    # 2. Train/test split (80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Train three ML algorithms
    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(random_state=42, n_estimators=100),
    }

    results = {}
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)
        results[name] = {"RMSE": rmse, "R2": r2}
        fitted_models[name] = model

    # 4. Compare performance
    results_df = pd.DataFrame(results).T.sort_values("RMSE")
    print("Model comparison:\n", results_df)

    # 5. Best-performing model (lowest RMSE)
    best_name = results_df.index[0]
    best_model = fitted_models[best_name]
    print(f"\nBest performing model: {best_name}")

    # 6. Save best model with joblib
    joblib.dump(best_model, "best_model.pkl")
    print("Saved best model to best_model.pkl")

    return results_df, best_name


# ---------------------------------------------------------------------------
# Question 6: SQL (8 Marks)
# ---------------------------------------------------------------------------
def question6_sql(df):
    section("QUESTION 6: SQL")

    conn = sqlite3.connect(":memory:")
    sql_df = df.drop_duplicates().dropna(subset=["Salary"]).copy()
    sql_df.to_sql("employees", conn, index=False, if_exists="replace")

    queries = {
        "1. All records": "SELECT * FROM employees LIMIT 10;",
        "2. Salary > 50000": "SELECT * FROM employees WHERE Salary > 50000;",
        "3. Descending by salary": "SELECT * FROM employees ORDER BY Salary DESC;",
        "4. Avg salary per department (GROUP BY)":
            "SELECT Department, AVG(Salary) AS avg_salary FROM employees GROUP BY Department;",
        "5. Count per department":
            "SELECT Department, COUNT(*) AS record_count FROM employees GROUP BY Department;",
        "6. Top 5 records": "SELECT * FROM employees LIMIT 5;",
        "7. SUM of salary": "SELECT SUM(Salary) AS total_salary FROM employees;",
    }

    for label, query in queries.items():
        print(f"\n-- {label} --")
        print(f"SQL: {query}")
        result = pd.read_sql_query(query, conn)
        print(result.head(10))

    conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv("employees.csv")

    question1_numpy()
    df = question2_pandas(df)
    question3_visualizations(df)
    question4_eda(df)
    question5_machine_learning(df)
    question6_sql(df)

    section("DONE - all 6 questions completed")
