import sys
from pathlib import Path
from src.data.validate_alpaca_schema import AlpacaSchemaValidator

if __name__ == "__main__":
    data_dir = Path("data/processed")
    english_tokens_path = Path("data/processed/english_tokens.json")
    validator = AlpacaSchemaValidator(english_tokens_path if english_tokens_path.exists() else None)
    reports = validator.validate_dir(data_dir)
    any_invalid = False
    for report in reports:
        print(f"\nFile: {report['file']}")
        if 'total' in report:
            print(f"  Total examples: {report['total']}")
            print(f"  Valid: {report['valid']}")
            print(f"  Invalid: {report['invalid']}")
            if report['invalid'] > 0:
                any_invalid = True
                for err in report['errors']:
                    print(f"    Example #{err['index']}: {err['errors']}")
        else:
            print(f"  Error: {report.get('error', 'Unknown error')}")
    if any_invalid:
        sys.exit(1)
    else:
        print("\nAll datasets passed Alpaca schema validation.")
