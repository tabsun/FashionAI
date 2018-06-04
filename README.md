# FashionAI - Clothe Keypoints Detection

> A keypoint detection scheme __based on OpenPose__.

## Introduction

    This is an experimental project for FashionAI competetion which is developed based on OpenPose(Cao et al).
    
    And for each clothe type (dress / skirt / outwear / trousers / blouse) we trained one model separately taking vgg19 network as initial weights for the first several layers. Basically each model need 2 days to train on a 8-gpu machine.
    
    

## Requirement

    tensorflow(1.4.1+)
    
    opencv
    
    scikit-image
    
    numpy
    
    ...

## Training
    
    I wrote a shell bash to facilitate the training procedure. To train a model, taking blouse as an example just set the --tag in ./begin_to_train.sh to 'blouse'.
    
    And then run ./begin_to_train.sh. 
    
    This command will start the training progress and the trained checkpoint models are saved under ./models/trained. Next you need to frozen the model to use it do the prediction.
    
    Just run ./begin_to_frozen.sh. 
    
    This command will generate frozen graph under ./tmp . 
    
    Copy it to the corresponding directory at ./models/trained/clothe_type/graph because the test shell will load model from this directory.
    
    ```shell
    >  cd code
    
    >  ./begin_to_train.sh
    
    >  ./begin_to_frozen.sh
    
    >  cp ./tmp/frozen_graph.pb ./models/trained/blouse/graph
    
    ```
    
    That's all !
    

## Test & Submit

    Here is another shell for test.
    
    First please make sure thers exits the corresponding frozen_graph.pb under "./models/trained/%s" % clothe_type.
    
    Modify the --tag in begin_to_test.sh and run it.
    
    This command will generate 'submit_%s.csv' % clothe_type under ./submit directory.
    
    After test all 5 clothe_types then `python merge.py` will merge all results into one .csv file which is in the submit csv format.
    
    '''shell
    
    >  cd code
    
    >  python run.py --model="cmu" --image="../data/test_b/test.csv" --tag="blouse" --test="submit" --resolution="368x368" --scales="[1.0, (0.5,0.25,1.5), (0.5,0.75,1.5), (0.25,0.5,1.5), (0.75,0.5,1.5), (0.5,0.5,1.5)]"

    >  python run.py --model="cmu" --image="../data/test_b/test.csv" --tag="dress" --test="submit" --resolution="368x368" --scales="[1.0, (0.5,0.25,1.5), (0.5,0.75,1.5), (0.25,0.5,1.5), (0.75,0.5,1.5), (0.5,0.5,1.5)]"

    >  python run.py --model="cmu" --image="../data/test_b/test.csv" --tag="skirt" --test="submit" --resolution="368x368" --scales="[1.0, (0.5,0.25,1.5), (0.5,0.75,1.5), (0.25,0.5,1.5), (0.75,0.5,1.5), (0.5,0.5,1.5)]"
    
    >  python run.py --model="cmu" --image="../data/test_b/test.csv" --tag="trousers" --test="submit" --resolution="368x368" --scales="[1.0, (0.5,0.25,1.5), (0.5,0.75,1.5), (0.25,0.5,1.5), (0.75,0.5,1.5), (0.5,0.5,1.5)]"

    >  python run.py --model="cmu" --image="../data/test_b/test.csv" --tag="outwear" --test="submit" --resolution="368x368" --scales="[1.0, (0.5,0.25,1.5), (0.5,0.75,1.5), (0.25,0.5,1.5), (0.75,0.5,1.5), (0.5,0.5,1.5)]"

    >  cd ../submit
    
    >  python merge.py 
    
    '''
    
    >  That's all!
    
## Contact

    If you have any questions about this project. then send an email to buptmsg@gmail.com to let me know.
