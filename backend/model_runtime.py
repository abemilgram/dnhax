"""Worker-owned resident AMB3R runtime with no fallback backend."""

from .models import model_config


class ModelRuntime:
    def __init__(self):
        self.amb3r = None

    @property
    def loads(self):
        return self.amb3r.loads if self.amb3r is not None else 0

    def unload(self):
        if self.amb3r is not None:
            self.amb3r.unload()

    def infer(self, images, device, progress):
        config = model_config()
        if self.amb3r is None:
            from .amb3r_runtime import Amb3rRuntime

            self.amb3r = Amb3rRuntime()
        return self.amb3r.infer(images, config, device, progress)


runtime = ModelRuntime()
