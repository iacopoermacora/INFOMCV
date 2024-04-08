from stanford40 import create_stanford40_splits
from HMDB51 import create_hmdb51_splits



# Train and test files for model 1
train_files, train_labels, test_files, test_labels = create_stanford40_splits()

# TODO: Read the files and labels from the augmented dataset and append them to these






# Train and test files for model 2, 3 and 4
keep_hmdb51 = ["clap", "climb", "drink", "jump", "pour", "ride_bike", "ride_horse", 
            "run", "shoot_bow", "smoke", "throw", "wave"]
train_files, train_labels, test_files, test_labels = create_hmdb51_splits(keep_hmdb51)