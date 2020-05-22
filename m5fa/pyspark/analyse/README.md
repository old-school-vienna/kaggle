## Analyse data using pyspark
### Docker
```
docker build -t myspark .
```
Add the second mount if original is a link
```
docker run -it \
    -e DATADIR=/opt/data \
    -v /home/wwagner4/prj/kaggle/m5fa/pyspark/analyse:/opt/project \
    -v /data/kaggle:/opt/data \
    myspark bash
```

### Data
Copy the original data from kaggle into data/original. They are too big for git.
```
kaggle: https://www.kaggle.com/c/m5-forecasting-accuracy/data
drive: https://drive.google.com/drive/folders/1koGlsp1fcV0JsPvIUtrFS1E-UhMHV2aa
```
