# Python Basics, Data Structures & Intro to Pandas/Numpy

## Overview
This assignment implements a set of small Python exercises that build from core data-structure practice (tuples, dicts, sorting) to light-weight data analysis with **Pandas** and **NumPy**. The final section explores a real dataset to extract simple descriptive insights.

## Repository Contents
- `python_basics.py` — Solutions and runnable entry points for all parts.
- `bank_additional_full.csv` — Bank Marketing dataset (tabular data used in Part 6).

## What’s Implemented

### Part 1 — Count occurrences of tuples in a list
Given a list of tuples, compute the frequency of each unique tuple and return a dictionary mapping `tuple -> count`. Useful patterns:
- Iteration over list elements
- Use of `list.count()` and/or dictionary accumulation

### Part 2 — Balanced BST from a list of random numbers
Generate 10 random integers in `[1, 100]`, sort them, and build a height-balanced Binary Search Tree by inserting the median repeatedly (divide-and-conquer). Demonstrates:
- Random number generation
- Recursion and mid-point splitting to approximate balance

### Part 3 — Dictionary key reconstruction from values
Given a mapping of words → letters, create a list of keys (words) that can be *formed using only the set of values* available in the dictionary. Highlights:
- Character set membership checks
- Iterating over keys and comparing against the value alphabet

### Part 4 — Sorting a list of tuples
For `listA = [(1,2), (4,3), (2,10), (12,5), (6,7), (9,11), (15,4)]`:
- Sort by **1st element** ascending
- Sort by **2nd element** ascending
- Repeat both sorts in **descending** order  
Illustrates use of `sorted()` with `key=lambda t: t[i]` and `reverse=True`.

### Part 5 — Mutable vs. Immutable (conceptual)
Clear notes and examples:
- **Immutable**: `str`, `tuple`
- **Mutable**: `list`, `dict`  
Explains why in-place operations work on mutable objects and produce new objects for immutable ones.

### Part 6 — Pandas & NumPy mini-analysis
Using `bank_additional_full.csv`, demonstrate common exploratory steps:

**Examples included**
- Unique values for a categorical column (e.g., `education`)
- Subscription counts (e.g., outcomes `y == "yes"` vs `y == "no"`)
- Group means of all independent variables by outcome `y`
- Mean `age` by `marital` status
- Null checks across columns
- Descriptive statistics (`.describe()`)

**Indexing & Querying**
- `pd.query()` for expressive filters
- `loc`/`iloc` for label/position based selection
- Range and boolean subsetting to extract up to **five** concise insights

> _Note_: The focus is on demonstrating reasoning with indexing and queries, not on model training. (Basic linear-regression and gradient-descent concepts are acknowledged at a conceptual level but not modeled here.)

## Getting Started

### 1) Environment
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt  # if present
# or just:
pip install pandas numpy

2) Running
python assignment_1.py


By default, the script runs each part’s demo code (printing results to stdout).
If you prefer running a specific part, you can call the corresponding function from an interactive session:

from assignment_1 import Assignment_1

# Part 1 example
sample = [(1,2), (1,2), (2,3)]
print(Assignment_1.part1(sample))

# Part 6 examples (requires CSV present in the repo)
Assignment_1.part6()  # prints summaries and insights

Sample Snippets

Unique values (education)

df['education'].unique()


Subscribed vs. not subscribed

df['y'].value_counts()        # or
(df['y'] == 'yes').sum(), (df['y'] == 'no').sum()


Means by outcome

df.groupby('y').mean(numeric_only=True)


Mean age by marital

df.groupby('marital')['age'].mean()


Null checks & describe

df.isna().sum()
df.describe(include='all')


Illustrative query/indexing

# Customers with tertiary education and age > 40
df.query('education == "tertiary" & age > 40')

# First 10 rows, selected columns by position
df.iloc[:10, :5]

# Rows where balance > 1500 and contacted in month of May
df.loc[(df['balance'] > 1500) & (df['month'] == 'may')]

Notes

Code favors clarity over micro-optimizations; it’s meant for learning.

The dataset should reside in the project root (same folder as the script).

If your console truncates Pandas output, adjust display options (e.g., pd.set_option('display.max_columns', None)).
