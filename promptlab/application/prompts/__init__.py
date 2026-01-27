from functools import lru_cache
from pathlib import Path

from jinja2 import Template

PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    prompt_file = PROMPTS_DIR / f"{name}.md"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_file}")

    return prompt_file.read_text()


def get_judge_suffix(score_min: int = 1, score_max: int = 10) -> str:
    template_str = load_prompt("judge_suffix")
    template = Template(template_str)
    return template.render(min=score_min, max=score_max)


def get_cot_prefix() -> str:
    return load_prompt("judge_cot_prefix")
