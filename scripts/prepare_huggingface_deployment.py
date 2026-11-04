"""Prepare, and optionally upload, the browser pipeline's Hugging Face services.

Default preparation makes no remote changes. --upload creates a private dataset
and a public, token-protected Docker API Space. Secrets are sent only through the
Hugging Face secrets API, never written to the staged source or printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def manifest_assets(folder: Path):
    folder = folder.resolve()
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("format") != "semantic-harmonization-minilm-v1":
        raise ValueError("Expected an exported MiniLM browser manifest")
    descriptors = [v["lexical"] for v in manifest["vocabularies"]]
    descriptors += [asset for v in manifest["vocabularies"] for chunk in v["chunks"]
                    for asset in (chunk["rows"], chunk["vectors"])]
    for asset in descriptors:
        if Path(asset["url"]).is_absolute() or ".." in Path(asset["url"]).parts:
            raise ValueError("Manifest assets must use relative paths within the export")
        source = (folder / asset["url"]).resolve()
        if not source.is_relative_to(folder) or not source.is_file():
            raise ValueError("A manifest asset points outside the export or is missing")
        if source.stat().st_size != asset["bytes"]:
            raise ValueError("Asset size does not match its manifest")
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != asset["sha256"]:
            raise ValueError("Asset hash does not match its manifest")
    return manifest, [asset["url"] for asset in descriptors]


def prepare(assets: Path, output: Path):
    manifest, paths = manifest_assets(assets)
    dataset = output / "dataset"
    space = output / "space"
    dataset.mkdir(parents=True, exist_ok=True)
    space.mkdir(parents=True, exist_ok=True)
    for relative in ["manifest.json", *paths]:
        source = assets / relative
        destination = dataset / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if os.path.samefile(source, destination):
                continue
            destination.unlink()
        try:
            os.link(source, destination)
        except OSError:
            shutil.copyfile(source, destination)
    (dataset / ".gitattributes").write_text("*.f32 filter=lfs diff=lfs merge=lfs -text\n")
    (dataset / "README.md").write_text(
        "---\npretty_name: Clinical terminology MiniLM browser packs\n---\n\n"
        "Precomputed MiniLM vectors and associated terminology descriptions for browser retrieval.\n\n"
        "Keep this repository private. Access to the source terminologies remains subject to their licenses. "
        "The classification Space streams these assets to authorized users; it does not build embeddings.\n"
    )
    docker = (ROOT / "backend/Dockerfile").read_text().replace("ENV PORT=8000", "ENV PORT=7860").replace("EXPOSE 8000", "EXPOSE 7860")
    (space / "Dockerfile").write_text(docker)
    (space / "README.md").write_text(
        "---\ntitle: Clinical Codelist API\nsdk: docker\napp_port: 7860\n---\n\n"
        "Token-protected classification and GPT review through OpenAI, plus streaming access to a private terminology dataset.\n\n"
        "Configure OPENAI_API_KEY, PIPELINE_API_TOKEN and HF_TOKEN as Space secrets. "
        "Configure PIPELINE_HF_DATASET and PIPELINE_ALLOWED_ORIGINS as variables. "
        "No terminology vectors are stored in the Space.\n"
    )
    code_paths = ["backend/app.py", "backend/requirements.txt"] + [str(path.relative_to(ROOT)) for path in sorted((ROOT / "src").rglob("*.py"))]
    for relative in code_paths:
        destination = space / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return {"dataset": dataset, "dataset_paths": ["manifest.json", ".gitattributes", "README.md", *paths],
            "space": space, "space_paths": ["README.md", "Dockerfile", *code_paths],
            "count": sum(v["count"] for v in manifest["vocabularies"])}


def upload(prepared, account, dataset_name, space_name):
    from huggingface_hub import HfApi, get_token
    from huggingface_hub.errors import RepositoryNotFoundError

    token = get_token()
    if not token:
        raise ValueError("Log in to the intended Hugging Face account first")
    for name in ("OPENAI_API_KEY", "PIPELINE_API_TOKEN"):
        if not os.getenv(name, "").strip() or os.environ[name].startswith("replace-"):
            raise ValueError(f"Set {name} in the environment before uploading")
    api = HfApi()
    profile = api.whoami()
    if account != profile["name"]:
        raise ValueError("The logged-in account does not match --account; switch accounts before uploading")
    dataset_id, space_id = f"{account}/{dataset_name}", f"{account}/{space_name}"
    try:
        existing_space = api.space_info(space_id)
        variable = api.get_space_variables(space_id).get("PIPELINE_HF_DATASET")
        if existing_space.private or existing_space.sdk != "docker" or variable is None or variable.value != dataset_id:
            raise ValueError("The target Space already exists and is not this public, token-protected API. Choose another Space name")
    except RepositoryNotFoundError:
        pass
    try:
        existing = api.repo_info(dataset_id, repo_type="dataset")
        if not existing.private:
            raise ValueError("The target dataset already exists publicly. Choose a private dataset name")
    except RepositoryNotFoundError:
        pass
    api.create_repo(dataset_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=dataset_id, repo_type="dataset", folder_path=prepared["dataset"],
                      allow_patterns=prepared["dataset_paths"], commit_message="Publish precomputed browser MiniLM packs")
    api.create_repo(space_id, repo_type="space", space_sdk="docker", space_hardware="cpu-basic", private=False, exist_ok=True)
    runtime_token = os.getenv("PIPELINE_HF_READ_TOKEN") or (token if profile.get("auth", {}).get("type") != "oauth" else None)
    runtime_secrets = {"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"], "PIPELINE_API_TOKEN": os.environ["PIPELINE_API_TOKEN"]}
    if runtime_token:
        runtime_secrets["HF_TOKEN"] = runtime_token
    for key, value in runtime_secrets.items():
        api.add_space_secret(repo_id=space_id, key=key, value=value)
    for key, value in {"PIPELINE_HF_DATASET": dataset_id, "PIPELINE_CLASSIFIER_PROVIDER": "openai",
                       "PIPELINE_ALLOWED_ORIGINS": "https://ruurdkuiper.github.io,http://127.0.0.1:5173"}.items():
        api.add_space_variable(repo_id=space_id, key=key, value=value)
    api.upload_folder(repo_id=space_id, repo_type="space", folder_path=prepared["space"],
                      allow_patterns=prepared["space_paths"], commit_message="Deploy OpenAI classification and protected terminology streaming")
    host = api.space_info(space_id).host
    if not host:
        raise RuntimeError("The Space was uploaded but has not returned a runtime URL yet")
    if not host.startswith("https://"):
        host = "https://" + host
    if not runtime_token:
        print("Dataset upload complete. Add a persistent read-only HF_TOKEN in the Space secrets to enable private downloads.")
    return {"apiBaseUrl": host.rstrip("/"), "manifestUrl": host.rstrip("/") + "/api/assets/manifest.json"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account", required=True)
    parser.add_argument("--dataset-name", default="semantic-harmonization-minilm")
    parser.add_argument("--space-name", default="semantic-harmonization-api")
    parser.add_argument("--assets-dir", type=Path, default=ROOT / "dist/browser-assets")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist/huggingface")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    for value in (args.account, args.dataset_name, args.space_name):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            parser.error("Use a Hugging Face account and repository names containing letters, digits, underscores or hyphens")
    load_dotenv(ROOT / ".env")
    prepared = prepare(args.assets_dir, args.output_dir)
    print(f"Prepared {prepared['count']:,} precomputed vectors and the OpenAI API Space. No secrets in staged files.")
    if args.upload:
        urls = upload(prepared, args.account, args.dataset_name, args.space_name)
        (args.output_dir / "urls.json").write_text(json.dumps(urls, indent=2))
        print("Classification API:", urls["apiBaseUrl"])
        print("Protected terminology manifest:", urls["manifestUrl"])
    else:
        print("No repositories created and no files uploaded. Use --upload after confirming the account and access settings.")


if __name__ == "__main__":
    main()
