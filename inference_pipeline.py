from transformers import pipeline, AutoTokenizer
import yaml
from dotmap import DotMap


def load_yaml(path) -> DotMap:
    with open(path, 'r') as f:
        file = yaml.safe_load(f)
        return DotMap(file)


class InferencePipeline():
    def __init__(self):
        self.config = load_yaml('settings/config.yaml')
        self.inference_pipe = self.load_model(self.config)

    def load_model(self, config):
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)

        pipe = pipeline("summarization", model=config.model_path,
                        tokenizer=tokenizer)
        return pipe
