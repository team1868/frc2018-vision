# Space Cookies Vision Object Detection 

Labeled cubes in photos to create data training set
Images can contain zero to a number of cubes in one frame
Training and validation, teaching the neural network what a cube is and giving it references where looking for cubes respectively
Training is 60% and validation is 40% of our dataset
Negative images (images without cubes) are used to differentiate power cubes and other yellow or cubic objects
Used images such as the FIRST logo, yellow storage boxes, white objects (overexposure may cause many false positives), black crates etc.  so the network is not recognizing cubes as any rectangular object or any yellow surface.
varying distances and amount occluded
About 6000 labelled images total, large datasets practical for a deep-learning neural network
NVIDIA Digits to train a neural network to recognize cubes
Digits easy to use for monitoring training performance as models develop
Inputting jpg images generated from videos by ffmpeg and txt files in KITTI format from various labelling programs including one written ourselves in python, using corresponding names ex: 00001.jpg and 00001.txt
Network will split the training photos into 500 parts, as we are setting the preferences to 500 epochs, then looks at each set of images to figure out what is and isn’t a cube
Can identify cubes regardless of scale, orientation, exposure, and surroundings using different sets of images
Have been tested in brightly lit or shadowed areas and cubes are still seen
If cubes are a moderate distance apart, over 4 inches, they will be individually recognized
Can tell if a cube is truncated, neural network will guess full coordinates of cube even if mostly offscreen
Can see cubes with about 70% visibility or more
Using NVIDIA Jetson TX2 for fast onboard processing
ZMQ (Zero Messaging Queues) to communicate with the RoboRio
Calculate the approximate distance and angle of the cube using the real cube dimensions and the size of the rectangle recognizing the cube.
Uses the largest seen bounding box, so the robot will go to the nearest cube
With logitech c615 can process at 19.3 flips per second at 640x480 resolution in live streaming without generating display; 19.1 flips per second at 800x600 resolution; 17.4 flips per second at 1280x720 resolution
Currently using 640x480 resolution because of speed and compatibility with CameraServer which is for streaming to the DriverStation
Object Detection used in autonomous mode so the robot can find and go to cubes


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you have to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [ZMQ](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc
