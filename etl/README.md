# ETL

Requirements

- python version > 3
- requirements.txt

How to run

```bash
python etl.py etl/config.ini | tee /srv/data/etl/`date '+%Y%m%d-%H%M'`_etl.log
```
