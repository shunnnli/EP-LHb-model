# ─── gymnasium 0.27+ compatibility shim ─────────────────────────────────────
import sys, os, types

try:
    # original import
    from gymnasium.wrappers.monitoring import video_recorder
except ImportError:
    # new RecordVideo wrapper
    from gymnasium.wrappers import RecordVideo

    class VideoRecorder(RecordVideo):
        """
        A drop‐in replacement for gym.wrappers.monitoring.video_recorder.VideoRecorder
        using the new Gymnasium RecordVideo API.
        """
        def __init__(self, *, env, base_path, metadata):
            # RecordVideo wants: env, video_folder, episode_trigger, name_prefix
            folder = os.path.dirname(base_path)
            prefix = os.path.basename(base_path)
            # record every frame once started
            super().__init__(
                env=env,
                video_folder=folder,
                episode_trigger=lambda step: True,
                name_prefix=prefix
            )
            # mimic the old attribute
            self.path = folder + os.sep + prefix + ".mp4"

        def capture_frame(self):
            # RecordVideo intercepts .step() and .render(), so no per‐frame call is needed
            # but we call render() once so it grabs the current frame
            if hasattr(self.env, "render"):
                self.env.render()

        def close(self):
            # finish writing out the video
            super().close()

    # inject it into the expected namespace
    video_recorder = types.SimpleNamespace(VideoRecorder=VideoRecorder)
    VecVideoRecorder = VideoRecorder
