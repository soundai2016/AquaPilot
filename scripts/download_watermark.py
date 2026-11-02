#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse

try:
    import requests
    from bs4 import BeautifulSoup
except ModuleNotFoundError as error:
    raise SystemExit("Install the download dependencies: python -m pip install -e '.[download]'") from error

OFFICIAL_PAGE = "https://www.ffi.no/en/research/watermark"


def resolve_download_url(session: requests.Session) -> str:
    response = session.get(OFFICIAL_PAGE, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    candidates: list[str] = []
    for anchor in soup.find_all("a", href=True):
        text = " ".join(anchor.get_text(" ", strip=True).split()).lower()
        href = urljoin(OFFICIAL_PAGE, anchor["href"])
        if "watermarkv1" in text or ("watermark" in href.lower() and href.lower().split("?")[0].endswith(".zip")):
            candidates.append(href)
    if not candidates:
        raise RuntimeError(
            "Could not locate the official WatermarkV1 ZIP link. "
            f"Open {OFFICIAL_PAGE} manually and verify that the page structure has not changed."
        )
    for candidate in candidates:
        host = urlparse(candidate).hostname or ""
        if host == "ffi.no" or host.endswith(".ffi.no"):
            return candidate
    raise RuntimeError("The page exposed candidates, but none were hosted by ffi.no; refusing third-party downloads.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download WatermarkV1 from the official FFI page")
    parser.add_argument("--output", type=Path, default=Path("data/raw/watermark"))
    parser.add_argument("--filename", default="WatermarkV1.zip")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="Resolve and validate the current official FFI attachment URL without downloading it",
    )
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": "AquaPilot"})
    if args.resolve_only:
        print(resolve_download_url(session))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / args.filename
    if destination.exists() and not args.force:
        print(f"Using existing file: {destination}")
    else:
        url = resolve_download_url(session)
        print(f"Resolved official FFI attachment: {url}")
        with session.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if "zip" not in content_type and not url.lower().split("?")[0].endswith(".zip"):
                print(f"Warning: unexpected content type {content_type!r}; validating ZIP signature after download.")
            temporary = destination.with_suffix(destination.suffix + ".part")
            digest = hashlib.sha256()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        digest.update(chunk)
            with temporary.open("rb") as handle:
                if handle.read(4) != b"PK\x03\x04":
                    temporary.unlink(missing_ok=True)
                    raise RuntimeError("Downloaded attachment does not have a ZIP signature.")
            temporary.replace(destination)
            print(f"SHA256: {digest.hexdigest()}")
            (args.output / "WatermarkV1.sha256").write_text(
                f"{digest.hexdigest()}  {destination.name}\n", encoding="utf-8"
            )

    if not args.no_extract:
        import zipfile

        extract_dir = args.output / "WatermarkV1"
        if extract_dir.exists() and any(extract_dir.iterdir()) and not args.force:
            print(f"Extraction directory already populated: {extract_dir}")
        else:
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(destination) as archive:
                archive.extractall(extract_dir)
            print(f"Extracted to: {extract_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
