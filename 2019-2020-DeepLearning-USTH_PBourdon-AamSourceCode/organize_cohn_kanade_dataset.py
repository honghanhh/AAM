# -*- coding: utf-8 -*-

import sys
import argparse
import os
import glob
import numpy as np
import shutil

def main(argv):
    parser = argparse.ArgumentParser(description='Cohn-Kanade dataset organizer : this program reads the original\
                                     Cohn-Kanade dataset from an input folder containing "images", "landmarks" and "emotion" subfolders and saves\
                                     useful data into an AAM analysis-friendly output directy')
    parser.add_argument('input_dir', help='input directory')
    parser.add_argument('output_dir', help='output directory')

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    print('*** Re-organizing Cohn-Kanade dataset ***')
    print('... input_dir : "{}"'.format(input_dir))
    print('... output_dir : "{}"'.format(output_dir))
    
    img_dir = os.path.join(input_dir, './images')
    assert os.path.exists(img_dir), 'Cannot find "images" subfolder'
    
    landmark_dir = os.path.join(input_dir, './landmarks')
    assert os.path.exists(landmark_dir), 'Cannot find "landmarks" subfolder'

    emotion_dir = os.path.join(input_dir, './emotion')
    assert os.path.exists(emotion_dir), 'Cannot find "emotion" subfolder'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for subject_dir in os.scandir(img_dir):
        for emotion_dir in os.scandir(subject_dir.path):
            if os.path.isdir(emotion_dir.path):
                search_pattern = os.path.join(emotion_dir.path,'S[0-9][0-9][0-9]_[0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].png')
                files = sorted(glob.glob(search_pattern))
                selected_files = np.array(files)[[0,-1]]    # selecting extreme poses
                for k, img_path in enumerate(selected_files):
                    landmarks_path = os.path.abspath(img_path)
                    landmarks_path = landmarks_path.replace('images', 'landmarks').replace('.png', '_landmarks.txt')
                    emotion_path = os.path.abspath(img_path)
                    emotion_path = emotion_path.replace('images', 'emotion').replace('.png', '_emotion.txt')

                    if os.path.exists(landmarks_path):
                        img_newpath = os.path.join(output_dir, os.path.basename(img_path))
                        landmarks_newpath = os.path.join(output_dir, os.path.basename(landmarks_path))
                        emotion_newpath = os.path.join(output_dir, os.path.basename(emotion_path))

                        if os.path.exists(emotion_path):
                            shutil.copy2(emotion_path, emotion_newpath)
                        elif k==0:      # neutral expressions have no .txt emotion file
                            with open(emotion_newpath, 'w') as f:
                                f.write('{:03.7e}'.format(0))
                        else:           # don't save data that has no emotion tag
                            continue
                        shutil.copy2(img_path, img_newpath)
                        shutil.copy2(landmarks_path, landmarks_newpath)
                
    print('***************** DONE! *****************')

if __name__ == '__main__':
    main(sys.argv)
