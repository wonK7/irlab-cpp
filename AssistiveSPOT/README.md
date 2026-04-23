# SPOT
This project is about using Boston Dynamics' SPOT robot to pick up a medicine bottle on a table. 

## Overview

The SPOT robot uses object detection to identify where the bottle is on the table and determines how to position the claw to grab it. 

## Searching for Object

Spot starts by deploying its arm and slowly rotating it in a 80 degree arc in front of itself. It repeats this motion at various heights. If it recognizes the medicine bottle, it stops, calculates the pose for the arm to move to, and then attempts to pick up the medicine bottle.

## Object Detection

A computer vision model is used to identify the game pieces, which includes recognizing the medicine bottle amid various other objects within the visual field of the robot. 

## 8/30 Improvements
 * Fixed grasp success logic so the arm doesn't keep cycling through poses once a grasp is successful.
 * Adjusted arm search height to improve detection accuracy when bottle is positioned on the back of the table. Height adjustment also fixed colliding with the table when attempting grasp.
 * Fixed grasp attempt logic to reduce cases of arm being stuck when medicine bottle not initially detected when attempting grasp.

## Future Work

SPOT sometimes takes a while to indentify the bottle, especially on a cluttered table, so I plan to retrain the computer vision model with more images of cluttered backgrounds to increase the accuracy and detection rate. I also hope to add more features like dropping off the medicine bottle and ways to detect the medicine bottle if it is obstructed from a front view.