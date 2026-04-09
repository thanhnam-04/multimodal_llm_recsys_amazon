import json
import logging
from pathlib import Path
from typing import Any

from src.evaluation.ablation_studies import AblationStudy
from src.utils.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def _ensure_test_with_responses_exists(processed_dir: Path) -> Path:
    """Ensure test_with_responses.json exists; create placeholders from test.json if needed."""
    responses_path = processed_dir / "test_with_responses.json"
    if responses_path.exists():
        return responses_path

    test_path = processed_dir / "test.json"
    if not test_path.exists():
        raise FileNotFoundError(
            "Missing both data/processed/test_with_responses.json and data/processed/test.json"
        )

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    for entry in test_data:
        entry.setdefault("model_response", "")

    with open(responses_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4)

    logger.warning(
        "test_with_responses.json was missing; created placeholder file from test.json "
        "with empty model_response values"
    )
    return responses_path


def _process_entries(data: list[dict[str, Any]], num_items: int) -> list[dict[str, Any]]:
    for entry in data:
        model_response = entry.get("model_response", "")

        if not model_response or model_response.strip() == "":
            model_response_items = ["<|endoftext|>"] * num_items
            logger.warning(f"Empty model response for entry, padded with {num_items} EOS tokens")
        else:
            model_response_items = [item.strip() for item in model_response.split(", ") if item.strip()]

            if len(model_response_items) < num_items:
                num_missing = num_items - len(model_response_items)
                model_response_items.extend(["<|endoftext|>"] * num_missing)
                logger.debug(f"Padded {num_missing} EOS tokens to reach {num_items} items")

            model_response_items = model_response_items[:num_items]

        entry["model_response_items"] = model_response_items
        entry["model_response"] = ", ".join(model_response_items)

    return data


def main() -> None:
    setup_logging()
    logger.info("Processing model outputs...")

    config = load_config()
    num_items = config["data_config"]["number_of_items_to_predict"]
    run_ablation = config["training_config"].get("ablation_studies", False)

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    source_path = _ensure_test_with_responses_exists(processed_dir)
    output_path = processed_dir / "test_with_responses_processed.json"

    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = _process_entries(data, num_items)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    logger.info(f"Model outputs processed successfully: {output_path}")

    if run_ablation:
        logger.info("Running ablation studies...")
        try:
            with open(processed_dir / "train.json", "r", encoding="utf-8") as f:
                train_data = json.load(f)

            with open(processed_dir / "test.json", "r", encoding="utf-8") as f:
                test_data = json.load(f)

            ablation_study = AblationStudy(config)
            ablation_results = ablation_study.run_complete_ablation(train_data, test_data, {})

            logger.info("Ablation studies completed successfully")
            logger.info(f"Best modality: {ablation_results['summary']['best_modality']}")
            logger.info(f"Best fusion method: {ablation_results['summary']['best_fusion_method']}")
        except Exception as e:
            logger.error(f"Ablation studies failed: {e}")
            logger.info("Continuing without ablation studies")
    else:
        logger.info("Ablation studies disabled")


if __name__ == "__main__":
    main()
