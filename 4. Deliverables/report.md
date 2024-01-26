# Title: A Student's Exploration into Agricultural Image Processing and Decision Making: My Journey with the Full_pipeline Script

## Introduction:
As a student deeply fascinated by the intersection of technology and agriculture, I embarked on an ambitious project to create the Full_pipeline script. This script is the culmination of my explorations into the realms of computer vision, deep learning, and reinforcement learning, all aimed at innovating in agricultural practices through advanced image analysis.

## The Genesis: Understanding the Basics with Import Statements
My journey with the pipeline began with laying the foundational blocks. I chose a suite of Python libraries, each catering to specific aspects of the script. cv2 and skimage for image manipulations, numpy and pandas for data handling, TensorFlow and Keras for deep learning, and stable_baselines3 for reinforcement learning. These tools were crucial in building a robust and versatile script.

## Early Milestones: Image Preprocessing and Patch Creation
The script's initial focus is on image preprocessing - a step I recognized as vital to ensure the uniformity of data. I apply various traditional cv techniques to cut all the images to the same size and exclude the unnecessary and even disturbing borders, I do this by looking at all connected components in the binary image and choosing the biggest one. Following this, I create and save image patches from those, before we can transform them we have to pad the images so that the image is dividable by the patches in both width and height, i did this with my own padder function. This step was crucial for handling large images by breaking them down into smaller, more manageable segments to stay in the bounds of my gpu memory and be able to pass them as batches.

## Deep Dive: Patch Analysis and Model Building
Next, I delved into the world of deep learning. I defined a weight map for my images, focusing on important regions within the image during model training, which is the middle. So i just defined the weightmap as a gradient which is lowest on the borders and highest in the middle, this was crucial to get the model to stop focusing on edges or other disturbances on the corners. Next i wrote my custom metrics classes F1 Score and IoU for model performance. For the model architecture i chose the U-Net with a ResNet50 backbone for effective image segmentation. Here i can import my weighted loss function with the map created earlier by adding a new function which calculates the binary_crossentropy and multiplicates it to all my weights from he map and then take the average, adding a layer of specificity to the learning process. Important is also to mention that i initialized my model with the weights of the pre-trained ResNet50 backbone, which is a common practice in deep learning. I then compile the model with the Adam optimizer and the custom metrics. Next up is to load the trained model for the later predictions on patches.

### Enhancing Model Specifics: The Power of ResNet50 and U-Net in Agricultural Image Processing
Building upon the earlier sections, it's important to highlight why the combination of ResNet50 and U-Net architectures forms a potent solution for our plant image processing task. ResNet50, known for its deep residual learning framework, excels in feature extraction even from very deep networks, which is important for accurately identifying the patterns in agricultural images. By integrating it with the U-Net architecture, renowned for its efficiency in image segmentation tasks, the model gains an exceptional ability to not only recognize but also precisely delineate our plant roots and leaves.
```python
def resnet_unet(input_size, num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_size, classes=1)
```

### Leveraging ResNet50 Layers:
In the construction of the ResNet50 U-Net model, different layers of the pre-trained ResNet50 serve as the backbone. By utilizing layers like conv4_block6_out and conv3_block4_out, the model can extract rich, hierarchical feature representations. This is particularly valuable in our task, where different scales of features (like minor root tips or larger plant structures) are essential for accurate analysis.
```python
    conv4 = base_model.get_layer('conv4_block6_out').output
    conv3 = base_model.get_layer('conv3_block4_out').output
    conv2 = base_model.get_layer('conv2_block3_out').output
    conv1 = base_model.get_layer('conv1_relu').output
```

### UpSampling and Concatenation:
The process of UpSampling and concatenating feature maps from ResNet50 with the U-Net architecture is a critical step. It ensures that the model does not just analyze high-level features but also retains finer details lost during downsampling.
```python
    up4 = UpSampling2D((2, 2))(conv4)
    up4 = concatenate([up4, conv3])
    up4 = Conv2D(256, (3, 3), activation='relu', padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Dropout(0.2)(up4)
    
    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([up3, conv2])
    up3 = Conv2D(128, (3, 3), activation='relu', padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Dropout(0.2)(up3)

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([up2, conv1])
    up2 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Dropout(0.2)(up2)

    up1 = UpSampling2D((2, 2))(up2)
    up1 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Dropout(0.2)(up1)
```

### Incorporation of Dropout and Batch Normalization:
My use of Dropout and Batch Normalization in each UpSampling step addresses two key challenges in deep learning: overfitting and internal covariate shift. Dropout reduces the model's reliance on any single pattern, promoting generalization, while Batch Normalization helps in stabilizing the learning process. This combination enhances the model's robustness, a necessary feature for the varied and unpredictable nature of our plant images.

### Sigmoid Activation for Final Layer:
Choosing a sigmoid activation function for the final layer aligns perfectly with the need for binary segmentation in our images, such as distinguishing between plant and non-plant elements.
```python
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)
    model = Model(inputs=base_model.input, outputs=outputs)
```

### Model Training:
During the training i tried many different architectures, also different backbones like the EfficientNet which performed a little worse. All of my initial models were multiclass because thats the logical thing to do when given multiple mask classes to detect. It turned out that we only needed a binary one with only the roots so i changed my entire model two days before the kaggle competition to be optimized to just roots but i ended up screwing up my own model with that and got a very bad placement on kaggle. I wanted to try some different models still like Inception but i didnt have enough time for that.

## Predictive Insights: Image Processing for Prediction
Now we get to the exciting part, predicting the masks for the image patches. When predicting the patches it is important to keep the order of the patches according to the original image. I then use my own written unpatchify function because the imported one is too much of a pain to use.

## Refinement and Detailing: Post-Processing of Predicted Masks
Post-prediction, the part were we cover up our models weakpoints. Here i am using binary threshholding and morphological operations to remove the noise and fill in the gaps and connect some roots. Another important step i implemented is to check for each connected component its pixel density since roots tend to be very thin and spread, this enables me to remove water droplets or other disturbances which are much higher in pixel density. Together with this i also checked its circularity just to be sure to remove unwanted objects. Using this i am left with only the roots and can then save the masks.

## A New Angle: Skeletonization and Analysis
Next up is the skeletonization which is a very important step to extract root ends and branch points. It was used to calculate root lengths as well but that is not needed in the pipeline. I then get all my endpoints for each skeleton and look for the lowest point in a skeleton, this leaves me with the main root ends. This is not the most accurate way but it is the most efficient way. I could have also used the same system which i used to identify the main root in my task 12 but it achieves the same results for the majority so i decided to trade off accuracy with efficency. I then get the main root ends and save them.
```python
def draw_main_root(image, skeleton_branch_data, u):
    info_image = np.copy(image).astype(np.uint8)  # Convert to uint8 if needed
    info_image *= 255
    main_root_lengths = {}
    lateral_root_lengths = {}

    for skeleton_id, group in skeleton_branch_data.groupby('skeleton-id'):
        G = nx.Graph()
        lateral_root_lengths[skeleton_id] = []
        
        # Add edges based on branch data, with actual lengths as weights
        for index, row in group.iterrows():
            src = (int(row['image-coord-src-1']), int(row['image-coord-src-0']))
            dst = (int(row['image-coord-dst-1']), int(row['image-coord-dst-0']))
            length = row['branch-distance']  # The actual length of the branch segment
            G.add_edge(src, dst, weight=length)

        endpoints = [node for node, degree in G.degree() if degree == 1]
        topmost_point = min(G.nodes, key=lambda point: point[1])
        max_path_length = 0
        # Find the longest path based on actual branch lengths
        for end_point in endpoints:
            if topmost_point != end_point:
                try:
                    length, path = nx.single_source_dijkstra(G, topmost_point, end_point, weight='weight')
                    if length > max_path_length:
                        longest_path = path
                        max_path_length = length
                except nx.NetworkXNoPath:
                    continue
        plant_id = assign_plant_id(topmost_point[0])
        main_root_lengths[plant_id] = max_path_length
        for edge in G.edges(data=True):
            if not (edge[0] in longest_path and edge[1] in longest_path):
                lateral_length = edge[2]['weight']
                lateral_root_lengths[skeleton_id].append(lateral_length)

```
This is how i get the root lengths, lateral root lengths, landmarks and endpoints. I am using all points from my summarization of the skeleton and load them into a graph with all the edges. Here it is important to add the branch lengths as weights so that it doesnt calculate the distance with the shortest path but rather the actual accurate root length. Next i am extracting the start point in the skeleton by looking at all nodes y axis and picking the highest (y axis is inverted so i search for the lowest y value). Next i am looking for all paths from this points until i find the longest possible, this will be my main root, important here again to add the weights for accurate lengths. This is how an image with landmarks looks:
![Comparison Graph](C:\Users\benjm\Documents\GitHub\2023-24-fai2-adsai-BenjaminGraziadei223946\4. Deliverables\landmarks.png)

## Bridging Digital and Physical: Point Extraction and Coordinate Transformation
Last part of the pipeline is the robotics part. Here i am using the coordinates of the root ends, which i transorm to the real world coordinates using my get_meter_coordinates function. This is how it works: i get the pixel coordinate on the screen convert those into meters with a given conversion rate (24pixels/mm) and then due to the robots workarea being much bigger than the space my plants are on i have to add the offset in the code set as start_point, important here is to swap the points x and y axis because the robots axis are inverted.

## The Climactic Integration: Reinforcement Learning Environment Interaction
In the final stage, I implemented my reinforcement learning. This was trained over multiple days with many different hyperparameters and reward functions. My reward function looks like this:
```python
def _calculate_reward(self, pipette_pos):
        cur_distance_to_goal = np.linalg.norm(pipette_pos - self.goal_position)
        prev_distance_to_goal = np.linalg.norm(self.prev_pipette_pos - self.goal_position) if self.prev_pipette_pos is not None else cur_distance_to_goal

        # Calculate improvement in distance
        distance_improvement = prev_distance_to_goal - cur_distance_to_goal

        # Define constants for rewards, penalties, and scaling factors
        GOAL_REACHED_REWARD = 200.0
        SCALER = 10
        

        # Update penalty/reward based on direction of movement
        if distance_improvement > 0:  # Moving towards the goal
            self.consecutive_wrong_direction = 0
            self.consecutive_right_direction += 1
            distance_reward = distance_improvement * self.consecutive_right_direction * SCALER
        else:  # Moving away from the goal
            self.consecutive_right_direction = 0
            self.consecutive_wrong_direction += 1
            distance_reward = distance_improvement * self.consecutive_wrong_direction * SCALER

        # Check if the goal is reached
        termination_threshold = 0.001
        if cur_distance_to_goal < termination_threshold:
            terminated = True
            reward = GOAL_REACHED_REWARD + distance_reward
        else:
            terminated = False
            reward = distance_reward

        # Check for truncation
        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False

        # Update the previous position for the next step
        self.prev_pipette_pos = pipette_pos.copy()

        return reward, terminated, truncated
```
Here i calculate the current distance from pipette to goal and the distance of the previous step and subtract them. If i got closer to the goal now this is a positive value of the right distance travelled and if he is going away from the goal it is a negative reward and also as big as he messed up. This assures that the agent tries to get the the goal as fast as possible. Also for reaching the goal he recieves an extra 200 points to assure thats the main goal. Lastly i implemented consecutive rights or wrongs so that the agent is motivated to to keep getting closer, because it is a linear increment of the reward he gets, same for negative. I performed my training on a remote server so that it could run independenty since this process takes multiple days and used weights and biases to keep track of my agents metrics and to save them later on. The remote server was also essential for the possibility to train multiple agent at the same time. Using my trained PPO model, I set out to make goal-oriented decisions based on the insights gained from image analysis. This part of the script marked the transition from computer vision to robotics.


Conclusion:
Reflecting on my journey with the Full_pipeline script, I feel a sense of accomplishment and excitement. This project was more than just a script; it was an exploration of how interdisciplinary approaches in technology can contribute significantly to agriculture. It stands as a testament to my learning journey and my aspirations to innovate in the field of agricultural technology.