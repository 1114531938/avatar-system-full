from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


@dataclass
class PipelineState:
    input_wav: str
    input_video: Optional[str] = None
    video_frames_dir: Optional[str] = None
    avatar_id: str = "306"
    base_name: str = ""

    run_id: str = ""
    run_dir: str = ""
    log_dir: str = ""

    prepare_only: bool = False
    launch_viewer: bool = True

    current_stage: str = "init"
    finished_stages: List[str] = field(default_factory=list)
    failed_stage: Optional[str] = None
    error: Optional[str] = None

    perception_json: Optional[str] = None
    task1_input_json: Optional[str] = None
    plan_json: Optional[str] = None
    selected_avatar_id: Optional[str] = None
    selected_tts_speaker_id: Optional[str] = None
    background: Optional[str] = None
    session_id: Optional[str] = None
    turn_id: Optional[str] = None

    task1_reply_json: Optional[str] = None
    reply_text: Optional[str] = None
    reply_style: Optional[str] = None
    tts_speaker_id: Optional[str] = None

    emotivoice_txt: Optional[str] = None
    reply_wav: Optional[str] = None

    deeptalk_npy: Optional[str] = None
    flame_motion_npz: Optional[str] = None

    point_cloud_path: Optional[str] = None
    template_npz: Optional[str] = None

    viewer_command: Optional[str] = None
    viewer_started: bool = False
    viewer_pid: Optional[int] = None

    artifact_dir: Optional[str] = None
    artifact_reply_wav: Optional[str] = None
    artifact_flame_motion_npz: Optional[str] = None
    output_video: Optional[str] = None
    output_white_model_video: Optional[str] = None
    artifact_enhanced_reply_wav: Optional[str] = None
    video_export_command: Optional[str] = None
    video_export_error: Optional[str] = None

    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
