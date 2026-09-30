import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Path routing relative to repository root
SFT_DOC_DIR = Path(__file__).resolve().parent
SRC_DIR = SFT_DOC_DIR.parent.parent
ROOT_DIR = SRC_DIR.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "sft_doc_config.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Loads configuration settings lazily from a YAML file.

    Pure function: Does not create directories or run side-effects on import.
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "The 'PyYAML' package is required to parse YAML configs. "
            "Please install dependencies via pip install pyyaml."
        ) from e

    if not config_path.exists():
        example_path = config_path.with_name("sft_doc_config.yaml.example")
        raise FileNotFoundError(
            f"Configuration file not found at: {config_path}\n"
            f"Please copy '{example_path}' to '{config_path}' and set your credentials."
        )

    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_file_content(file_path: Path) -> str:
    """Utility function to read raw text files safely."""
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found at: {file_path}")
    return file_path.read_text(encoding="utf-8").strip()


def clean_markdown_json(raw_response: str | None) -> str:
    """Strips markdown code fences and returns clean raw JSON string safely.

    Pure function: No network or filesystem dependencies.
    """
    if not raw_response:
        raise ValueError("Received empty or None response from model API.")

    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_response.strip()


def check_batch_shape(parsed: object) -> None:
    """Raises ValueError unless parsed is an object whose "records" is a list of objects.

    Pure function: No network or filesystem dependencies.
    """
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}.")
    records = parsed.get("records", [])
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        raise ValueError('Expected "records" to be a list of objects.')


def parse_batch_response(raw_response: str | None) -> dict:
    """Parses a model reply into a batch dict whose "records" is a list of objects.

    Raises ValueError, which generate_batch retries, for valid JSON of the wrong shape.
    Pure function: No network or filesystem dependencies.
    """
    parsed = json.loads(clean_markdown_json(raw_response))
    check_batch_shape(parsed)
    return parsed


def assign_hash_ids(parsed_batch: dict) -> dict:
    """Generates unique, collision-proof deterministic IDs using prompt content hashes.

    Pure function: No network or filesystem dependencies.
    """
    domain_map = {
        "poultry_production": "poul",
        "swine_husbandry": "swin",
        "veterinary_pharmacology": "vetp",
        "crop_agronomy": "crop",
    }

    type_map = {
        "uncritical_industrial": "ind",
        "control_neutral": "neu",
        "control_crop": "cro",
    }

    for record in parsed_batch.get("records", []):
        raw_domain = record.get("domain", "")
        domain_code = domain_map.get(raw_domain, raw_domain[:4] if raw_domain else "gen")

        task = record.get("functional_task", "adv")[:3]
        type_code = type_map.get(record.get("data_type"), "unk")

        prompt_str = record.get("prompt", "") + record.get("assistant_response", "")
        content_hash = hashlib.md5(prompt_str.encode("utf-8")).hexdigest()[:8]

        record["id"] = f"{domain_code}-{task}-{type_code}-{content_hash}"

    return parsed_batch


def generate_batch(
    prompt_path: Path | None = None,
    context_path: Path | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
    max_retries: int = 3,
) -> dict:
    """Executes the extraction call with explicit timeouts and automatic retries.

    Explicit parameters are honored first (including temperature=0.0). Missing
    parameters are lazily resolved from configs/sft_doc_config.yaml.
    """
    try:
        from openai import APIError, OpenAI
    except ImportError as e:
        raise ImportError("The 'openai' package is required to run batch generation.") from e

    # Check if ANY explicit parameter is missing using explicit 'is None' checks
    missing_args = [
        prompt_path is None,
        context_path is None,
        api_key is None,
        base_url is None,
        model is None,
        temperature is None,
    ]

    # Only attempt config loading if at least one parameter needs fallback resolution
    if any(missing_args):
        cfg = load_config(config_path)
        api_cfg = cfg.get("api", {})
        def_cfg = cfg.get("defaults", {})

        api_key = api_key or api_cfg.get("openrouter_api_key")
        base_url = base_url or api_cfg.get("openrouter_base_url")
        model = model or api_cfg.get("teacher_model")

        if temperature is None:
            temperature = def_cfg.get("temperature", 0.3)

        if prompt_path is None:
            prompt_path = ROOT_DIR / def_cfg.get(
                "system_prompt_path", "configs/prompts/docgen_system_prompt.md"
            )

        if context_path is None:
            context_path = ROOT_DIR / def_cfg.get("context_path", "context/excerpt.txt")

    system_instructions = load_file_content(prompt_path)
    source_context = load_file_content(context_path)

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=180.0,
        default_headers={
            "HTTP-Referer": "https://github.com/kairos-strategic/em-sft-benchmark",
            "X-Title": "Fragile Compassion SFT Data Generator",
        },
    )

    user_payload = f"### SOURCE EXCERPT FOR THIS RUN\n{source_context}"

    for attempt in range(1, max_retries + 1):
        print(f"Sending extraction request via {model} (Attempt {attempt}/{max_retries})...")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": user_payload},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )

            parsed_json = parse_batch_response(response.choices[0].message.content)
            record_count = len(parsed_json.get("records", []))
            print(f"Successfully generated and parsed {record_count} records.")
            return parsed_json

        except (APIError, ValueError, json.JSONDecodeError) as e:
            print(f"Error on attempt {attempt}: {e}")
            if attempt < max_retries:
                sleep_time = attempt * 5
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("Max retries reached. Generation failed.")
                raise e


def consolidate_output_directory(output_dir: Path) -> dict:
    """Scans all JSON batch files in output_dir, deduplicates records by ID, and exports
    master_dataset.json.

    Pure filesystem operations; does not require secrets or API keys.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    master_records = {}
    master_file_path = output_dir / "master_dataset.json"

    json_files = list(output_dir.glob("*.json"))
    batch_file_count = 0

    for file in json_files:
        if file.name == "master_dataset.json":
            continue

        batch_file_count += 1
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            check_batch_shape(data)
        except ValueError as e:  # includes json.JSONDecodeError
            print(f"Warning: Could not parse {file.name} ({e}). Skipping.")
            continue

        for record in data.get("records", []):
            rec_id = record.get("id")
            if rec_id:
                master_records[rec_id] = record

    consolidated_data = {"records": list(master_records.values())}

    master_file_path.write_text(json.dumps(consolidated_data, indent=2), encoding="utf-8")

    print("\n--- CONSOLIDATION SUMMARY ---")
    print(f"Processed batch files: {batch_file_count}")
    print(f"Unique records in master dataset: {len(master_records)}")
    print(f"Master file updated at: {master_file_path}")

    return consolidated_data


def main():
    """CLI execution entrypoint."""
    cfg = load_config()
    output_dir = ROOT_DIR / cfg.get("defaults", {}).get("output_dir", "output")
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_results = assign_hash_ids(generate_batch())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_filename = f"batch_{timestamp}.json"
    batch_file_path = output_dir / batch_filename

    batch_file_path.write_text(json.dumps(batch_results, indent=2), encoding="utf-8")
    print(f"Saved current run to: {batch_file_path}")

    consolidate_output_directory(output_dir=output_dir)


if __name__ == "__main__":
    main()
