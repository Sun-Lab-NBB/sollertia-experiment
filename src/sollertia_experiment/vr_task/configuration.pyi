from dataclasses import dataclass

from sollertia_shared_assets import TaskTemplate

@dataclass(frozen=True, slots=True)
class VRTaskConfiguration:
    ip: str = ...
    port: int = ...

def load_vr_task_template(unity_scene_name: str) -> TaskTemplate: ...
