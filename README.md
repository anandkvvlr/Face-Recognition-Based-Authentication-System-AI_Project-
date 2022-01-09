# Face-Recognition-Based-Authentication-System (AI_Project)

![image](https://user-images.githubusercontent.com/88075268/148682917-8e0e3b4a-f5b0-4aa2-ab7c-ef18b48a5235.png)

### Project Goals
#### Objective 
      To Build an AI based Face Recognition Model which Recognize the face and gives the access to the user
![image](https://user-images.githubusercontent.com/88075268/148683082-69190399-833f-42ba-b2a4-cec0f549079c.png)

      
### Constraints
      - The user needs to be in a well lit are and at a distance which the camera can focus for face recognition to work
      - Data collected using face recognition can require large amounts of storage
      
### Project Overview
      - The objective of the project is to capture the face to recognize the user and grant access, we have used Insight face and Retina Face-R50 to achieve that
      - Facial recognition is a means of identifying or confirming an individual’s identity using their face. Facial    recognition systems can be used to identify people in               photos, videos or in real-time.
      - Implementation of face recognition includes the following stages:
      - Data acquisition (capturing faces of multiple people)
      - Data storage (storing the facial images captured in the database)
      - Data processing (compare the images captured with the images in the database)
      - Decision making (grant access if the image matches with any image in the database else save the image to the database)
      - Present the user with an option to enter the username and password if the image matches else present the user the option to sign up/register to the service
      - The model frames the face within a box and shows the name of the user below the face
      - We use a for loop to capture multiple faces

