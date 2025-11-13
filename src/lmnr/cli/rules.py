from pathlib import Path
import urllib.request
import urllib.error
import sys


from lmnr.sdk.log import get_default_logger

LOG = get_default_logger(__name__)


def add_cursor_rules() -> None:
    """Download laminar.mdc file from a hardcoded public URL and save it to .cursor/rules/laminar.mdc"""
    # Hardcoded URL for the laminar.mdc file
    url = "https://raw.githubusercontent.com/lmnr-ai/lmnr/dev/rules/laminar.mdc"

    # Create .cursor/rules directory if it doesn't exist
    rules_dir = Path(".cursor/rules")
    rules_dir.mkdir(parents=True, exist_ok=True)

    # Define the target file path
    target_file = rules_dir / "laminar.mdc"

    try:
        LOG.info(f"Downloading laminar.mdc from {url}")

        # Download the file
        with urllib.request.urlopen(url) as response:
            content = response.read()

        # Write the content to the target file (this will overwrite if it exists)
        with open(target_file, "wb") as f:
            f.write(content)

        LOG.info(f"Successfully downloaded laminar.mdc to {target_file}")

    except urllib.error.URLError as e:
        LOG.error(f"Failed to download file from {url}: {e}")
        sys.exit(1)
    except Exception as e:
        LOG.error(f"Unexpected error: {e}")
        sys.exit(1)
