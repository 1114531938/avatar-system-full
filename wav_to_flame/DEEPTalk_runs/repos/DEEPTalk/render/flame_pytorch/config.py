import argparse

parser = argparse.ArgumentParser(description="FLAME model")

parser.add_argument(
    "--flame_model_path",
    type=str,
    default="../models/flame_models/generic_model.pkl",
    help="flame model path",
)

parser.add_argument(
    "--static_landmark_embedding_path",
    type=str,
    default="../models/flame_models/flame_static_embedding.pkl",
    help="Static landmark embeddings path for FLAME",
)

parser.add_argument(
    "--dynamic_landmark_embedding_path",
    type=str,
    default="../models/flame_models/flame_dynamic_embedding.npy",
    help="Dynamic contour embedding path for FLAME",
)

# FLAME hyper-parameters

parser.add_argument(
    "--shape_params", type=int, default=100, help="the number of shape parameters"
)

parser.add_argument(
    "--expression_params",
    type=int,
    default=50,
    help="the number of expression parameters",
)

parser.add_argument(
    "--pose_params", type=int, default=6, help="the number of pose parameters"
)

# Training hyper-parameters

parser.add_argument(
    "--use_face_contour",
    default=True,
    type=bool,
    help="If true apply the landmark loss on also on the face contour.",
)

parser.add_argument(
    "--use_3D_translation",
    default=True,  # Flase for RingNet project
    type=bool,
    help="If true apply the landmark loss on also on the face contour.",
)

parser.add_argument(
    "--optimize_eyeballpose",
    default=True,  # False for For RingNet project
    type=bool,
    help="If true optimize for the eyeball pose.",
)

parser.add_argument(
    "--optimize_neckpose",
    default=True,  # False For RingNet project
    type=bool,
    help="If true optimize for the neck pose.",
)

parser.add_argument("--num_worker", type=int, default=4, help="pytorch number worker.")

parser.add_argument("--batch_size", type=int, default=8, help="Training batch size.")

parser.add_argument("--ring_margin", type=float, default=0.5, help="ring margin.")

parser.add_argument(
    "--ring_loss_weight", type=float, default=1.0, help="weight on ring loss."
)

# custom arguments
# parser.add_argument("-image_dir", type=str, default="./images/base", help="dir for saving images")
parser.add_argument("--output_video_dir", type=str, default="/home/MIR_LAB/EMOTE/rend", help="output_video_dir")
parser.add_argument("--input_param_path", type=str, default="/home/MIR_LAB/EMOTE/flame_param", help="path for params")
parser.add_argument("--with_shape", type=bool, default=False, help="whether to use shape")
parser.add_argument("--with_shape_each", type=bool, default=False, help="whether to use shape")
parser.add_argument("--with_jaw", type=bool, default=False, help="whether to use jaw")
parser.add_argument("--with_global_rot", type=bool, default=False, help="whether to use global rotation")
parser.add_argument("--with_neck", type=bool, default=False, help="whether to use neck")
# parser.add_argument("--dataset", type = str, default = 'MEAD', help = "MEAD, RAVDESS,RAVDESS_separate")
parser.add_argument("--use_savgol_filter", type = bool, default = False, help = "whether to use savgol filter")
parser.add_argument("--use_bilateral_filter", type = bool, default = False, help = "whether to use bilateral filter")
# parser.add_argument("--add_audio", type = bool, default = False, help = "whether to add audio")
parser.add_argument("--audio_path", type = str, default = '', help = "audio path")
parser.add_argument("--output_image_path", type = str, default = '', help = "image path")
parser.add_argument("--output_vertices_path", type = str, default = '', help = "vertices path")
parser.add_argument("--original_video_path", type = str, default = '', help = "original video path")
parser.add_argument("--width", type = int, default = 800, help = "width of the video")
parser.add_argument("--height", type = int, default = 800, help = "height of the video")
parser.add_argument("--fps", type = int, default = 25, help = "fps of the video")
def get_config():
    config = parser.parse_args()
    return config
