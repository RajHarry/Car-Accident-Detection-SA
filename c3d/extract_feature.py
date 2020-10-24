import os
from c3d.c3d import c3d_feature_extractor,preprocess_input
from c3d.classifier import build_classifier_model
from c3d.utils.visualization_util import get_video_clips,visualize_predictions
import c3d.parameters as params
import c3d.configuration as cfg
from c3d.plot_controller import savitzky_golay
from c3d.utils.array_util import interpolate,extrapolate
from keras import backend as K 
import keras
import numpy as np

def extract_feature_video(video_path, progress_recorder, features_per_bag = params.features_per_bag):
    if keras.backend.backend() == 'tensorflow':
        K.clear_session()

    # read video
    video_clips, num_frames = get_video_clips(video_path)

    print("Number of clips in the video : ", len(video_clips))

    # build models
    feature_extractor = c3d_feature_extractor()
    classifier_model = build_classifier_model()

    print("Models initialized")

    # extract features
    rgb_features = []
    for i, clip in enumerate(video_clips):
        clip = np.array(clip)
        if len(clip) < params.frame_count:
            continue

        progress_recorder.set_progress(i, len(video_clips))
        clip = preprocess_input(clip)
        rgb_feature = feature_extractor.predict(clip)[0]
        rgb_features.append(rgb_feature)

        print("Processed clip : ", i)
        # break
    rgb_features = np.array(rgb_features)
    
    # bag features
    rgb_feature_bag = interpolate(rgb_features, features_per_bag)

    # classify using the trained classifier model
    predictions = classifier_model.predict(rgb_feature_bag)
    # assert(False)
    predictions = np.array(predictions).squeeze()
    predictions = extrapolate(predictions, num_frames)

    predictions = savitzky_golay(predictions, 101, 3)
    video_name = video_path.split("/")[-1].split(".")[0]
    # print("video_name(before_save): ",video_name)
    # np.save('media/features/{}'.format(video_name), predictions)

    save_path = os.path.join(cfg.output_folder, video_name + '.gif')
    # # visualize predictions
    visualize_predictions(video_path, predictions, save_path)
    return predictions

def load_npy(file_path):
    return np.load(file_path)

# extract_feature_video('media/RoadAccidents310_x264.mp4', 32)