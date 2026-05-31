# Retail Sales Analytics Dashboard

An interactive analytics dashboard built with Python and Plotly Dash. It analyses weekly retail sales data across 45 stores covering 2010 to 2012. The dashboard lets you explore sales trends, store performance, holiday impacts, and predictive model results through live filters and charts.

---

## What It Does

- Shows monthly and seasonal sales trends across all stores
- Compares store performance by type and size
- Breaks down holiday sales uplift by event (Christmas, Thanksgiving, Super Bowl, Labor Day)
- Displays Random Forest model predictions vs actual sales
- Shows feature importance from the trained model
- Lets you filter everything by store, year, store type, and holiday weeks
- Includes a searchable and sortable data explorer table

---

## Dashboard Tabs

| Tab | What You See |
|---|---|
| Sales Overview | Monthly trend, seasonal bar chart, top departments |
| Store Performance | Revenue rankings, size vs sales scatter, monthly heatmap |
| Holiday Analysis | Event comparison, markdown timing by week |
| Predictive Insights | Feature importance, actual vs predicted chart |
| Data Explorer | Live searchable and sortable records table |

---

## Dataset

The dashboard uses three CSV files from the [Kaggle Retail Dataset](https://www.kaggle.com/datasets/manjeetsingh/retaildataset):

| File | Rows | Description |
|---|---|---|
| `sales_data-set.csv` | 421,570 | Weekly sales by store and department |
| `stores_data-set.csv` | 45 | Store type and floor size |
| `Features_data_set.csv` | 8,190 | Weekly economic data and promotional markdowns |

Download the files from Kaggle and place them in the same folder as `dashboard.py`.

---

## Requirements

Python 3.9 or later is required.

Install the dependencies:

```bash
pip install dash plotly dash-bootstrap-components scikit-learn pandas
```

---

## How to Run

1. Clone this repo or download `dashboard.py`
2. Place all three CSV files in the same folder as `dashboard.py`
3. Run the dashboard:

```bash
python dashboard.py
```

4. Open your browser and go to:

```
http://127.0.0.1:8050
```

The dashboard will load automatically. The first load takes about 30 to 60 seconds because it trains the Random Forest model on startup.

---

## Project Structure

```
.
├── dashboard.py          # Main dashboard application
├── sales_data-set.csv    # Sales records (download from Kaggle)
├── stores_data-set.csv   # Store details (download from Kaggle)
└── Features_data_set.csv # Economic and promotional features (download from Kaggle)
```

---

## How It Works

**Data preparation** happens automatically on startup. The three CSV files are merged, missing values are handled, negative sales rows are removed, and new features like week number and total markdown are added.

**The predictive model** is a Random Forest trained on 2010 to 2011 data and evaluated on 2012 data. It achieves an R squared score of 0.92 on the test set. Department is the most important feature at 66 percent importance, followed by store size at 19 percent.

**Filters** at the top of the dashboard control all five tabs at once. You can filter by store, year, store type, or toggle to holiday weeks only.

---

## Tech Stack

- Python 3.12
- Plotly Dash
- Dash Bootstrap Components
- Plotly Express and Graph Objects
- pandas
- scikit-learn
- NumPy

---

## Notes

- The CSV files are not included in this repo. Download them from Kaggle using the link above.
- The model trains on a 30 percent subsample of training data for speed. You can increase this in the code by changing the `n_samples` parameter in the `resample` call.
- All data is processed in memory. Nothing is written to disk except the model predictions shown in the dashboard.

---

## License

MIT
