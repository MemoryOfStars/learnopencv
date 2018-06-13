import cv2
import sys
import numpy as np

major_ver, minor_ver, subminor_ver = cv2.__version__.split('.')

if __name__ == '__main__' :

    # Set up tracker.
    # Instead of MIL, you can also use

    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[1]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()


    # Read video
    video = cv2.VideoCapture("videos/chaplin.mp4")

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    
    # Define an initial bounding box
    bbox = (287, 23, 86, 320)

    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame,False)
    height = bbox[2]
    width = bbox[3]
    
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    
    #Initialize a canvas for depicting frames
    canvas = np.zeros((500,500,3),dtype="uint8")
    cv2.imshow("Frame",canvas)
    
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
        
        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)
        imTrimmed = frame[int(bbox[1]):int(bbox[1] + width), int(bbox[0]):int(bbox[0]+height)]
        
        #print(bbox[3] - bbox[1])
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (1,1,1), 2, 1)
            canvas[int(250-width/2):int(250 + width/2), int(250-height/2):int(250+height/2)] = imTrimmed
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
    
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

        #cv2.resizeWindow("Tracking",500,500)
        # Display result
        #cv2.rectangle(canvas,(200,50),(225,125),(255,0,0),-1)
        cv2.imshow("Canvas", canvas)
        cv2.imshow("imTrimmed",imTrimmed)
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
    #cv2.rectangle(canvas,(200,50),(225,125),(255,0,0),-1)
    #cv2.imshow("Canvas", canvas)
