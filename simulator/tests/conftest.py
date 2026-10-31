import pytest

from saltsim.runtime import SimRuntime
from saltsort_common.scene_config import SceneConfig, load_scene_config


@pytest.fixture(scope="session")
def scene_cfg() -> SceneConfig:
    return load_scene_config()


@pytest.fixture
def runtime(scene_cfg) -> SimRuntime:
    return SimRuntime(scene_cfg)
