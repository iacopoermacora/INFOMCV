# MODELS TO DEVELOP

# 1. Stanford 40 – Frames: Create a CNN and train it on the images in Stanford 40. Naturally, you will have 12 output classes.

# 2. HMDB51 – Frames (transfer learning): Use your pretrained CNN (same architecture/weights) and fine-tune it on the middle 
#    frame of videos of the HMDB51 dataset. You can use a different learning rate than for the Stanford 40 network training.

# 3. HMDB51 – Optical flow: Create a new CNN and train it on the optical flow of videos in HMBD51. You can use the middle frame
#    (max 5 points) or stack a fixed number (e.g., 16) of optical flow frames together (max 10 points).

# 4. HMDB51 – Two-stream: Finally, create a two-stream CNN with one stream for the frames and one stream for the optical flow. 
#    Use your pre-trained CNNs to initialize the weights of the two branches. Think about how to fuse the two streams and motivate 
#    this in your report. Look at the Q&A at the end of this assignment. Fine-tune the network.