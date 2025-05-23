import time
import os
import cv2

from sequence_utils import VOTSequence
from particle_filter import Particle_filter
#, MSParams
all_failures = 0
all_fps = []
names = [n for n in os.listdir("vids/")]
names = ["basketball"]
for name in names:
    # set the path to directory where you have the sequences
    dataset_path = 'vids' # TODO: set to the dataet path on your disk
    sequence = name  # choose the sequence you want to test

    # visualization and setup parameters
    win_name = 'Tracking window'
    reinitialize = True

    show_gt = False 
    video_delay = 15  
    font = cv2.FONT_HERSHEY_PLAIN   
  
    # create sequence object
    sequence = VOTSequence(dataset_path, sequence)
    init_frame = 0
    n_failures = 0
    # create parameters and tracker objects
    # sigmas = [0.3, 0.5, 0.7, 1.2]
    # distances = [0.05, 0.1, 0.5, 0.1]
    # weights = [1e-3, 1e-5, 1e-7]
    # alphas = [0, 0.01, 0.05, 0.1]

    n_bins = [8, 16, 20]
    # parameters = MSParams(1, 0.1, 40, 0, 20)
    tracker = Particle_filter(q_mult=0.1, dynamic_model="NCV")
    #parameters = MSParams()
    #tracker = MeanShiftTracker(parameters)

    time_all = 0

    # initialize visualization window
    sequence.initialize_window(win_name)
    # tracking loop - goes over all frames in the video sequence
    frame_idx = 0
    while frame_idx < sequence.length():
        img = cv2.imread(sequence.frame(frame_idx))
        # initialize or track
        if frame_idx == init_frame:
            # initialize tracker (at the beginning of the sequence or after tracking failure)
            t_ = time.time()
            tracker.initialize(img, sequence.get_annotation(frame_idx, type='rectangle'))
            time_all += time.time() - t_
            predicted_bbox = sequence.get_annotation(frame_idx, type='rectangle')
        else:
            # track on current frame - predict bounding box
            t_ = time.time()
            predicted_bbox = tracker.track(img)
            time_all += time.time() - t_

        # calculate overlap (needed to determine failure of a tracker)
        gt_bb = sequence.get_annotation(frame_idx, type='rectangle')
        o = sequence.overlap(predicted_bbox, gt_bb)

        # draw ground-truth and predicted bounding boxes, frame numbers and show image
        if show_gt:
            sequence.draw_region(img, gt_bb, (0, 255, 0), 1)
        sequence.draw_region(img, predicted_bbox, (0, 0, 255), 2)

        for particle in tracker.particles:
        # Draw particle position at (xc, yc)
            xc, yc = particle[0], particle[1]
            cv2.circle(img, (int(xc), int(yc)), 2, (0, 0, 255), -1)
        sequence.draw_text(img, '%d/%d' % (frame_idx + 1, sequence.length()), (25, 25))
        sequence.draw_text(img, 'Fails: %d' % n_failures, (25, 55))
        
        sequence.show_image(img, video_delay)

        if o > 0 or not reinitialize:
            # increase frame counter by 1
            frame_idx += 1
        else:
            # increase frame counter by 5 and set re-initialization to the next frame
            frame_idx += 5
            init_frame = frame_idx
            n_failures += 1

    print('Tracking speed: %.1f FPS' % (sequence.length() / time_all))
    print('Tracker failed %d times' % n_failures)
    all_failures += n_failures
    all_fps.append((sequence.length() / time_all))

print(f"Total fails: {all_failures}")
print(sum(all_fps)/len(all_fps))

