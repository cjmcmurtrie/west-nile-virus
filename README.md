# West Nile Disease

Best score achieved from this codebase was 0.81244 public, 0.79615 private.

## Requirements

- Python 3.5

See requirements.txt for Python package requirements.

## Usage

Create a `data/` directory in project root. Download and extract [this](https://drive.google.com/open?id=1UvwcmnLmOJejn_KQUhisUDJEou29m_P1) archive to the directory (there are extra data files not found in the Kaggle archive). Then run the following commands.

```
$ virtualenv -p python3.5 env/
$ source env/bin/activate
(env)$ pip install -r requirements.txt
(env)$ mkdir submissions/
(env)$ python src/ffn/run.py
```

A new timestamped submission file should be found in `submissions/`.
