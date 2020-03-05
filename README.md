# Fast_Image_Classification

Code for https://towardsdatascience.com/a-step-by-step-tutorial-to-build-and-deploy-an-image-classification-api-95fa449f0f6a

### Data
Download data : <https://github.com/CVxTz/ToyImageClassificationDataset>

### Docker

```sudo docker build -t img_classif .```

```sudo docker run -p 8080:8080 img_classif```

```time curl -X POST "http://127.0.0.1:8080/scorefile/" -H  "accept: application/json" -H  "Content-Type: multipart/form-data" -F "file=@337px-Majestic_Twilight.jpg"```
