# Fast_Image_Classification

Code for https://towardsdatascience.com/a-step-by-step-tutorial-to-build-and-deploy-an-image-classification-api-95fa449f0f6a

### Data
Download data : <https://github.com/CVxTz/ToyImageClassificationDataset>

### Docker

```sudo docker build -t img_classif .```

```sudo docker run -p 8080:8080 img_classif```

```time curl -X POST "http://127.0.0.1:8080/scorefile/" -H  "accept: application/json" -H  "Content-Type: multipart/form-data" -F "file=@337px-Majestic_Twilight.jpg"```

## Description

Our objective in this small project is to build an image classification API from scratch.
We are going to go through all the steps needed to achieve this objective :

* Data annotation (with Unsplash API + Labelme )

* Model Training ( With Tensorflow )

* Making the API ( With Uvicorn and FastApi )

* Deploying the API on a remote server ( With Docker and Google Cloud Platform )

## Data Annotation :

One of the most important parts of any machine learning project is the quality and quantity of the annotated data. It is one of the key factors that will influence the quality of the predictions when the API is deployed.

In this project we will try to classify an input image into four classes :

* City

* Beach

* Sunset

* Trees/Forest

I choose those classes because it is easy to find tons of images representing them online. We use those classes to define a multi-label classification problem :

![Examples of inputs and targets / Images from [https://unsplash.com/](https://unsplash.com/)](https://cdn-images-1.medium.com/max/2000/1*buGA2Qk4KXqJMq5Xu5gffg.png)*Examples of inputs and targets / Images from [https://unsplash.com/](https://unsplash.com/)*

Now that we have defined the problem we want to solve we need to get a sufficient amount of labeled samples for training and evaluation. 
To do that we will first use the [Unsplash](https://unsplash.com/) API to get URLs of images given multiple search queries.

    # First install [https://github.com/yakupadakli/python-unsplash](https://github.com/yakupadakli/python-unsplash)
    # Unsplash API [https://unsplash.com/documentation](https://unsplash.com/documentation)
    import json
    import os

    from unsplash.api import Api
    from unsplash.auth import Auth

    with open('tokens.json', 'r') as f:
        data = json.load(f)

    client_id = data['client_id']
    client_secret = data['client_secret']

    redirect_uri = ""
    code = ""

    keyword = 'beach'

    auth = Auth(client_id, client_secret, redirect_uri, code=code)
    api = Api(auth)

    photos = api.search.photos(keyword, per_page=1000, page=i)['results']

    for photo in photos:
        print(photo)
        print(photo.id)
        print(photo.urls)
        print(photo.urls.small)

We would try to get image URLs that are related to our target classes plus some other random images that will serve as negative examples.

The next step is to go through all the images and assign a set of labels to each one of them, like what is shown in the figure above. For this it is always easier to use annotations tools that are designed for this task like [LabelMe,](https://github.com/wkentaro/labelme) it is a python library that you can run easily from the command line:

    labelme . -flags labels.txt

![Labelme user interface](https://cdn-images-1.medium.com/max/2000/1*4iIY7gR7ZYEc-qEW3HJmLg.png)*Labelme user interface*

Using Labelme I labeled around a thousand images and made the urls+labels available here: [https://github.com/CVxTz/ToyImageClassificationDataset](https://github.com/CVxTz/ToyImageClassificationDataset)

## Model

Now that we have the labeled samples we can try building a classifier using Tensorflow. We will use MobileNet_V2 as the backbone of the classifier since it is fast and less likely to over-fit given the tiny amount of labeled samples we have, you can easily use it by importing it from keras_applications :

    from tensorflow.keras.applications import MobileNetV2

    base_model = MobileNetV2(include_top=False, input_shape=input_shape, weights=weights)

Since it is a multi-label classification problem with four classes, we will have an output layer of four neurons with the Sigmoid activation ( given an example, we can have multiple neurons active or no neuron active as the target)

### Transfer learning

One commonly used trick to tackle the lack of labeled samples is to use transfer learning. It is when you transfer some of the weights learned from a source task ( like image classification with a different set of labels) to your target task as the starting point of your training. This allows for a better initialization compared to starting from random and allows for reusing some of the representations learned on the source task for our multi-label classification.

Here we will transfer the weights that resulted from training in ImageNet. Doing this is very easy when using Tensorflow+Keras for MobileNet_V2, you just need to specify weights=”imagenet” when creating an instance of MobileNetV2

    base_model = MobileNetV2(include_top=False, input_shape=input_shape, weights="imagenet")

### Data augmentation

Another trick to improve performance when having a small set of annotated samples is to do data augmentation. It is the process of applying random perturbations that preserve the label information ( a picture of a city after the perturbations still looks like a city ). Some common transformations are vertical mirroring, salt and pepper noise or blurring.

![Data augmentation examples / Image from [https://unsplash.com/](https://unsplash.com/)](https://cdn-images-1.medium.com/max/3710/1*BNhj5p5uTfwF9yaeRDuBUQ.png)*Data augmentation examples / Image from [https://unsplash.com/](https://unsplash.com/)*

To achieve this we use a python package called imgaug and define a sequence of transformation along with their amplitude :

    sometimes = **lambda **aug: iaa.Sometimes(0.1, aug)
    seq = iaa.Sequential(
        [
            sometimes(iaa.Affine(scale={**"x"**: (0.8, 1.2)})),
            sometimes(iaa.Fliplr(p=0.5)),
            sometimes(iaa.Affine(scale={**"y"**: (0.8, 1.2)})),
            sometimes(iaa.Affine(translate_percent={**"x"**: (-0.2, 0.2)})),
            sometimes(iaa.Affine(translate_percent={**"y"**: (-0.2, 0.2)})),
            sometimes(iaa.Affine(rotate=(-20, 20))),
            sometimes(iaa.Affine(shear=(-20, 20))),
            sometimes(iaa.AdditiveGaussianNoise(scale=0.07 * 255)),
            sometimes(iaa.GaussianBlur(sigma=(0, 3.0))),
        ],
        random_order=**True**,
    )

### Training

We split the dataset into two folds, training and validation and use the binary_crossentropy as our target along with the binary_accuracy as the evaluation metric.

We run the training from the command line after updating some configuration files :

    # data_config.yaml for defnining the classes and input size**
    input_shape**: [null, null, 3]
    **resize_shape**: [224, 224]
    **images_base_path**: **'../example/data/'
    targets**: [**'beach'**, **'city'**, **'sunset'**, **'trees'**]
    **image_name_col**: **'name'**

    # training_config.yaml for defining some training parameters**
    use_augmentation**: true
    **batch_size**: 32
    **epochs**: 1000
    **initial_learning_rate**: 0.0001
    **model_path**: **"image_classification.h5"**

Then running the training script :

    **export PYTHONPATH=$PYTHONPATH:~/PycharmProjects/FastImageClassification/**

    **python train.py --csv_path "../example/data.csv" \
           --data_config_path "../example/data_config.yaml" \
           --training_config_path "../example/training_config.yaml"**

![](https://cdn-images-1.medium.com/max/2048/1*azNZV0IqtYNajUzboMaYDw.gif)

We end up with a validation binary accuracy of **94%**

## Making the API

We will be using FastAPI to expose a predictor through an easy to use API that can take as input an image file and outputs a JSON with the classification scores for each class.

First, we need to write a Predictor class that can easily load a tensorflow.keras model and have a method to classify an image that is in the form of a file object.

    **class **ImagePredictor:
        **def **__init__(
            self, model_path, resize_size, targets, pre_processing_function=preprocess_input
        ):
            self.model_path = model_path
            self.pre_processing_function = pre_processing_function
            self.model = load_model(self.model_path)
            self.resize_size = resize_size
            self.targets = targets

        @classmethod
        **def **init_from_config_path(cls, config_path):
            **with **open(config_path, **"r"**) **as **f:
                config = yaml.load(f, yaml.SafeLoader)
            predictor = cls(
                model_path=config[**"model_path"**],
                resize_size=config[**"resize_shape"**],
                targets=config[**"targets"**],
            )
            **return **predictor

        @classmethod
        **def **init_from_config_url(cls, config_path):
            **with **open(config_path, **"r"**) **as **f:
                config = yaml.load(f, yaml.SafeLoader)

            download_model(
                config[**"model_url"**], config[**"model_path"**], config[**"model_sha256"**]
            )

            **return **cls.init_from_config_path(config_path)

        **def **predict_from_array(self, arr):
            arr = resize_img(arr, h=self.resize_size[0], w=self.resize_size[1])
            arr = self.pre_processing_function(arr)
            pred = self.model.predict(arr[np.newaxis, ...]).ravel().tolist()
            pred = [round(x, 3) **for **x **in **pred]
            **return **{k: v **for **k, v **in **zip(self.targets, pred)}

        **def **predict_from_file(self, file_object):
            arr = read_from_file(file_object)
            **return **self.predict_from_array(arr)

We can use a configuration file to instantiate a predictor object that has all the parameters to do predictions and will download the model from the GitHub repository of the project :

    **# config.yaml
    resize_shape**: [224, 224]
    **targets**: [**'beach'**, **'city'**, **'sunset'**, **'trees'**]
    **model_path**: **"image_classification.h5"
    model_url**: **"https://github.com/CVxTz/FastImageClassification/releases/download/v0.1/image_classification.h5"
    model_sha256**: **"d5cd9082651faa826cab4562f60e3095502286b5ea64d5b25ba3682b66fbc305"**

After doing all of this, the main file of our API becomes trivial when using FastAPI :

    **from **fastapi **import **FastAPI, File, UploadFile

    **from **fast_image_classification.predictor **import **ImagePredictor

    app = FastAPI()

    predictor_config_path = **"config.yaml"**

    predictor = ImagePredictor.init_from_config_url(predictor_config_path)
    

    @app.post(**"/scorefile/"**)
    **def **create_upload_file(file: UploadFile = File(...)):
        **return **predictor.predict_from_file(file.file)

We can now run the app with a single command :

    uvicorn main:app --reload

This gives us access to Swagger UI where we can try out our API on a new file.

![[http://127.0.0.1:8080/docs](http://127.0.0.1:8000/docs)](https://cdn-images-1.medium.com/max/2898/1*ka58fs87P-ptDS8aciP4Gw.png)*[http://127.0.0.1:8080/docs](http://127.0.0.1:8000/docs)*

![Photo by [Antonio Resendiz](https://unsplash.com/@antonioresendiz_?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/city?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)](https://cdn-images-1.medium.com/max/5606/1*uPdQuS0Y4Esz5vYZk13Npw.jpeg)*Photo by [Antonio Resendiz](https://unsplash.com/@antonioresendiz_?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/city?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)*

Uploading the image above gives up the following output :

    {**
      **"beach":** 0**,**
      **"city":** 0**.**999**,**
      **"sunset":** 0**.**005**,**
      **"trees":** 0
    **}

Which is the expected output!

We can also send a request via curl and time it :

    time curl -X POST "[http://127.0.0.1:8080/scorefile/](http://127.0.0.1:8000/scorefile/)" -H  "accept: application/json" -H  "Content-Type: multipart/form-data" -F "file=[@antonio](http://twitter.com/antonio)-resendiz-VTLqQe4Ej8I-unsplash.jpg;type=image/jpeg"

    >> {"beach":0.0,"city":0.999,"sunset":0.005,"trees":0.0}
    >> real 0m0.209s
    >> user 0m0.012s
    >> sys 0m0.008s

## Deploying the App

### Docker

Its easier to deploy an app if it is inside a container like Docker.

We will create a Dockerfile with all the instructions needed to run our app after installing the correct environment :

    **FROM** python:3.6-slim
    **COPY** app/main.py /deploy/
    **COPY** app/config.yaml /deploy/
    **WORKDIR** /deploy/
    **RUN** apt update
    **RUN** apt install -y git
    **RUN** apt-get install -y libglib2.0-0
    **RUN** pip install git+https://github.com/CVxTz/FastImageClassification
    **EXPOSE** 8080

    **ENTRYPOINT** uvicorn main:app --host 0.0.0.0 --port 8080

Install Docker :

    sudo apt install docker.io

Then we can run the Docker build :

    sudo docker build -t img_classif .

We finally run the container while mapping the port 8080 of container to that of the host :

    sudo docker run -p 8080:8080 img_classif .

### Deploying on a remote server

I tried to do this on an ec2 instance from AWS but the ssh command line was clunky and the terminal would freeze at the last command, no idea why. So I decided to do the deployment using Google Cloud Platform’s App engine. Link to a more detailed tutorial on the subject [here](https://blog.machinebox.io/deploy-docker-containers-in-google-cloud-platform-4b921c77476b).

* Create a google cloud platform account

* install gcloud

* create project project_id

* clone [https://github.com/CVxTz/FastImageClassification](https://github.com/CVxTz/FastImageClassification) and call :

    cd FastImageClassification

    gcloud config set project_id

    gcloud app deploy app.yaml -v v1

The last command will take a while but … Voilaaa!

![](https://cdn-images-1.medium.com/max/3702/1*UcHnFytWaAIpeM-ZL_3Dng.png)

## Conclusion

In this project, we built and deployed machine-learning powered image classification API from scratch using Tensorflow, Docker, FastAPI and Google Cloud Platform‘s App Engine. All those tools made the whole process straightforward and relatively easy. The next step would be to explore questions related to security and performance when handling a large number of queries.
