import json
import glob
import argparse
from pathlib import Path
from string import Template

ROOT_DIR = Path(__file__).parent.parent

def get_tool_list(category: str) -> list[str]:
    train_data_dir = ROOT_DIR / "finetune" / category / "results" / "trajectories" / "train" / "tool_adaptors"
    tool_json_paths = glob.glob(f"{train_data_dir.resolve()}/*.jsonl")
    tool_list = []
    for tool_json_path in tool_json_paths:
        if category in tool_json_path:
            tool_list.append(Path(tool_json_path).stem)
    return tool_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True, help="Name of the category to generate template for")
    parser.add_argument("--model_folder", type=str, required=True, help="Folder of the base model containing chat templates template")
    args = parser.parse_args()

    model_folder = Path(args.model_folder)

    with open(model_folder / "tool_template.jinja", "r") as f:
        template_str = f.read()
    tool_list = get_tool_list(args.category)

    output_dir = model_folder / args.category
    output_dir.mkdir(parents=True, exist_ok=True)
    for tool in tool_list:
        filled = Template(template_str).substitute(TARGET_TOOL=tool)
        print(tool)
        with open(output_dir / f"{tool}.jinja", "w") as f:
            f.write(filled)
    print(f"Generated templates for category {args.category} in {output_dir}")
