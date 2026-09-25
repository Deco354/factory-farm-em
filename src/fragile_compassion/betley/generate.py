from datetime import datetime
import hashlib
import json
import re
import time
from pathlib import Path
from openai import OpenAI, APIError

# Path routing relative to repository root
BETLEY_DIR = Path(__file__).resolve().parent
SRC_DIR = BETLEY_DIR.parent.parent
ROOT_DIR = SRC_DIR.parent
CONFIGS_DIR = ROOT_DIR / "configs"

# Attempt to load local active config; fallback to config template error
try:
    import sys
    sys.path.append(str(CONFIGS_DIR))
    import config
except ImportError:
    raise ImportError(
        "Missing active configuration file. Please copy 'configs/config.py.example' "
        "to 'configs/config.py' and fill in your OpenRouter API credentials."
    )


def load_file_content(file_path: Path) -> str:
    """Utility function to read raw text files safely."""
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found at: {file_path}")
    return file_path.read_text(encoding="utf-8").strip()


def clean_markdown_json(raw_response: str | None) -> str:
    """Strips markdown code fences and returns clean raw JSON string safely."""
    if not raw_response:
        raise ValueError("Received empty or None response from model API.")

    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_response.strip()


def generate_batch(
    prompt_path: Path = config.DEFAULT_SYSTEM_PROMPT_PATH,
    context_path: Path = config.DEFAULT_CONTEXT_PATH,
    max_retries: int = 3,
) -> dict:
    """Executes the extraction call with explicit timeouts and automatic retries."""
    system_instructions = load_file_content(prompt_path)
    source_context = load_file_content(context_path)

    client = OpenAI(
        base_url=config.OPENROUTER_BASE_URL,
        api_key=config.OPENROUTER_API_KEY,
        timeout=180.0,
        default_headers={
            "HTTP-Referer": "https://github.com/kairos-strategic/em-sft-benchmark",
            "X-Title": "Fragile Compassion SFT Data Generator",
        },
    )

    user_payload = f"### SOURCE EXCERPT FOR THIS RUN\n{source_context}"

    for attempt in range(1, max_retries + 1):
        print(f"Sending extraction request via {config.TEACHER_MODEL} (Attempt {attempt}/{max_retries})...")

        try:
            response = client.chat.completions.create(
                model=config.TEACHER_MODEL,
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": user_payload},
                ],
                temperature=config.DEFAULT_TEMPERATURE,
                response_format={"type": "json_object"},
            )

            raw_content = response.choices[0].message.content
            cleaned_json_str = clean_markdown_json(raw_content)

            parsed_json = json.loads(cleaned_json_str)
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


def assign_hash_ids(parsed_batch: dict) -> dict:
    """Generates unique, collision-proof deterministic IDs using prompts content hashes."""
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
        domain_code = domain_map.get(
            raw_domain, raw_domain[:4] if raw_domain else "gen"
        )

        task = record.get("functional_task", "adv")[:3]
        type_code = type_map.get(record.get("data_type"), "unk")

        prompt_str = record.get("prompts", "") + record.get("assistant_response", "")
        content_hash = hashlib.md5(prompt_str.encode("utf-8")).hexdigest()[:8]

        record["id"] = f"{domain_code}-{task}-{type_code}-{content_hash}"

    return parsed_batch


def consolidate_output_directory(output_dir: Path = config.OUTPUT_DIR) -> dict:
    """Scans all JSON batch files in output_dir, deduplicates records by ID, and exports master_dataset.json."""
    master_records = {}
    master_file_path = output_dir / "master_dataset.json"

    json_files = list(output_dir.glob("*.json"))

    for file in json_files:
        if file.name == "master_dataset.json":
            continue

        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            records = data.get("records", [])

            for record in records:
                rec_id = record.get("id")
                if rec_id:
                    master_records[rec_id] = record
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {file.name}. Skipping.")

    consolidated_data = {"records": list(master_records.values())}

    master_file_path.write_text(
        json.dumps(consolidated_data, indent=2), encoding="utf-8"
    )

    print("\n--- CONSOLIDATION SUMMARY ---")
    print(f"Processed batch files: {len(json_files) - (1 if master_file_path.exists() else 0)}")
    print(f"Unique records in master dataset: {len(master_records)}")
    print(f"Master file updated at: {master_file_path}")

    return consolidated_data


def main():
    batch_results = assign_hash_ids(generate_batch())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_filename = f"batch_{timestamp}.json"
    batch_file_path = config.OUTPUT_DIR / batch_filename

    batch_file_path.write_text(
        json.dumps(batch_results, indent=2), encoding="utf-8"
    )
    print(f"Saved current run to: {batch_file_path}")

    consolidate_output_directory()


if __name__ == "__main__":
    main()