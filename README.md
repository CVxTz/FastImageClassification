# Fast_Image_Classification

### Data
Download data : <https://github.com/CVxTz/ToyImageClassificationDataset>

### Docker

```sudo docker build -t img_classif .```

```sudo docker run -p 80:80 img_classif```

```time curl -X POST "http://127.0.0.1:80/scorefile/" -H  "accept: application/json" -H  "Content-Type: multipart/form-data" -F "file=@337px-Majestic_Twilight.jpg"```