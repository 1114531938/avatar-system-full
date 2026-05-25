import os 

# if 'DISPLAY' not in os.environ or os.environ['DISPLAY'] == '':
#     os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ['PYOPENGL_PLATFORM'] =  'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np 
import trimesh 
import pyrender
import cv2
from flame_pytorch import FLAME, get_config
from PyRenderMeshSequenceRenderer import PyRenderMeshSequenceRenderer, get_vertices_from_FLAME,save_video
import torch
import tqdm
import subprocess
import json
radian = np.pi / 180.0

def get_video_dimensions(file_path):
    ffprobe_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'json',
        file_path
    ]

    result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
    json_data = json.loads(result.stdout)

    width = json_data['streams'][0]['width']
    height = json_data['streams'][0]['height']

    return int(width), int(height)
    

if __name__ == '__main__':
    config = get_config()
    # os.makedirs(config.image_dir, exist_ok=True)
    os.makedirs(config.output_video_dir, exist_ok=True)

    file_name = config.input_param_path.split('/')[-1].split('.')[0]
    out_video_path = os.path.join(config.output_video_dir, f'{file_name}.mp4')

    print('initializing renderer...')
    template_mesh_path = '/home/whwjdqls99/EMOTE/models/flame_models/FLAME_sample.ply'
    width = config.width
    height = config.height
    renderer = PyRenderMeshSequenceRenderer(template_mesh_path,
                                            bg_color="white",
                                            width=width,
                                            height=height)
    
    print('getting flame model and calculating vertices...')
    params = np.load(config.input_param_path)
    flame = FLAME(config, batch_size = len(params)).cuda()
    predicted_vertices = get_vertices_from_FLAME(flame, params, config.with_shape)
    
    print('rendering...')
    T = len(predicted_vertices)
    pred_images = []
    for t in tqdm.tqdm(range(T)):
        pred_vertices = predicted_vertices[t].detach().cpu().view(-1,3).numpy()
        pred_image = renderer.render(pred_vertices)
        pred_images.append(pred_image) 
    
    
    print('saving video...')
    pred_images = np.stack(pred_images, axis=0)  
    save_video(out_video_path, pred_images, fourcc="mp4v", fps=config.fps)
    
    if config.audio_path:
        print('adding audio...')
        # audio_path = os.path.join(config.audio_path, f'{file_name}.wav')
        audio_path = config.audio_path
        command = f'ffmpeg -y -i {out_video_path} -i {audio_path} -c:v copy -c:a aac -strict experimental {out_video_path.split(".mp4")[0]}_audio.mp4'
        os.system(command)
        os.remove(out_video_path)
    
    elif config.original_video_path:
        ori_width, ori_height = get_video_dimensions(config.original_video_path)
        print(ori_width, ori_height)
        if ori_width < width or ori_height < height:
            operation = 'increase'
        else :
            operation = 'decrease'
            
        print('copying original video...')
        temp_video_name = config.original_video_path.split('/')[-1].split('.')[0] + '_temp.mp4'
        temp_video_path = os.path.join(os.path.dirname(out_video_path), temp_video_name)
        command = f"ffmpeg -i {config.original_video_path} -vf 'scale={width}:{height}:force_original_aspect_ratio={operation},crop={width}:{height}' -c:a copy {temp_video_path}"
        os.system(command)
        
        print('stacking rendered video and original video...')
        command = f"ffmpeg -i {out_video_path} -i {temp_video_path} -filter_complex hstack=inputs=2 {out_video_path.split('.mp4')[0]}_stacked.mp4"
        os.system(command)
        
        # os.remove(temp_video_path)