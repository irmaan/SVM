## SVM implemetation for Handwritten Recognition

The program has been implemented in two ways, once the training operation has been done as One Vs Rest (ovr) and once as One Vs One (ovs). 
Also, there are two Python files in the model folder, one is a personal implementation and the other is an implementation with the scikit library.
The program has been implemented as Multi Class from the beginning, but the Linear SVM function has also been added.
Student's distribution was used to evaluate the fitted predictors.
Comments has been added to functions to explain the process.

## Data:

- MNIST dataset will be downloaded automatically using mnist library. You should install mnist using pip.

## Results:

### My implemenation:
- One By One Accuracy (train): 96.4
- One Vs One Accuracy (test) : 94.2
- One Vs Rest Accuracy (train): 97.8
- One Vs Rest Accuracy (test) : 95.6

### SKlearn:

- One By One Accuracy (train): 98.9
- One Vs One Accuracy (test) : 96.8
- One Vs Rest Accuracy (train): 98.6
- One Vs Rest Accuracy (test) : 96.6

