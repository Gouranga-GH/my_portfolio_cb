import argparse
import base64
import os
import re
import sys
import time
from typing import Dict, Iterable, List, Optional

import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Iterable as _Iterable

# Optional progress bar support
try:
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover - fallback when tqdm not available
    _tqdm = None

GITHUB_API_BASE = "https://api.github.com"


def build_request(url: str, token: Optional[str]) -> urllib.request.Request:
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "gh-projects-fetcher"
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def http_get_json(url: str, token: Optional[str]) -> Dict:
    req = build_request(url, token)
    with urllib.request.urlopen(req) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        data = resp.read().decode(charset)
        return json.loads(data)


def list_public_repositories(owner: str, token: Optional[str], include_forks: bool) -> List[Dict]:
    repos: List[Dict] = []
    page = 1
    while True:
        url_user = f"{GITHUB_API_BASE}/users/{owner}/repos?per_page=100&page={page}"
        url_org = f"{GITHUB_API_BASE}/orgs/{owner}/repos?per_page=100&page={page}"
        try:
            page_data = http_get_json(url_user, token)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                page_data = http_get_json(url_org, token)
            else:
                raise
        if not isinstance(page_data, list) or len(page_data) == 0:
            break
        repos.extend(page_data)
        page += 1
    if not include_forks:
        repos = [r for r in repos if not r.get("fork", False)]
    return repos


def fetch_readme_text(owner: str, repo: str, token: Optional[str]) -> Optional[str]:
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/readme"
    try:
        req = build_request(url, token)
        with urllib.request.urlopen(req) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            data = resp.read().decode(charset)
            payload = json.loads(data)
            if isinstance(payload, dict) and payload.get("encoding") == "base64" and payload.get("content"):
                return base64.b64decode(payload["content"]).decode("utf-8", errors="replace")
            if isinstance(payload, dict) and payload.get("download_url"):
                req_raw = build_request(payload["download_url"], token)
                with urllib.request.urlopen(req_raw) as raw_resp:
                    return raw_resp.read().decode("utf-8", errors="replace")
            return None
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            return None
        raise


def extract_topmost_title(markdown_text: Optional[str]) -> str:
    if not markdown_text:
        return ""
    for line in markdown_text.splitlines():
        m = re.match(r"^\s*#\s+(.+?)\s*$", line)
        if m:
            return m.group(1).strip()
    lines = markdown_text.splitlines()
    for i in range(len(lines) - 1):
        if re.match(r"^=+$", lines[i + 1].strip()):
            return lines[i].strip()
    return ""


def write_project_file(directory: str, repo_name: str, project_link: str, title: str, readme_text: Optional[str]) -> None:
    os.makedirs(directory, exist_ok=True)
    safe_repo_name = re.sub(r"[^A-Za-z0-9_.-]", "_", repo_name)
    file_path = os.path.join(directory, f"{safe_repo_name}.md")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Title: {title or repo_name}\n")
        f.write(f"Link: {project_link}\n\n")
        f.write("README:\n")
        if readme_text:
            f.write(readme_text)
        else:
            f.write("(No README found)\n")


def get_project_root() -> Path:
    """Return the repository root based on this file's location.

    Falls back to current working directory if path depth is insufficient.
    """
    try:
        # Given current layout: <root>/utils/scripts/fetch_github_projects.py
        return Path(__file__).resolve().parents[2]
    except IndexError:
        return Path.cwd()


def load_dotenv(env_path: Path) -> None:
    """Minimal .env loader (no external dependency).

    Sets environment variables only if they are not already defined.
    Supports optional 'export ' prefix and quoted values.
    """
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        # Silently ignore .env parsing errors to avoid breaking the CLI
        pass


def fetch_and_write(owner: str, out_dir: str, token: Optional[str], include_forks: bool, delay_seconds: float) -> None:
    repositories = list_public_repositories(owner, token, include_forks)

    # Prepare iterable with optional progress bar
    repo_iter: _Iterable[Dict]
    if _tqdm is not None:
        repo_iter = _tqdm(
            repositories,
            desc=f"Processing repos for {owner}",
            unit="repo",
            ncols=100,
            leave=False,
        )
    else:
        repo_iter = repositories

    for repo in repo_iter:
        repo_name = repo.get("name", "")
        project_link = repo.get("html_url", "")
        readme_text = fetch_readme_text(owner, repo_name, token)
        title = extract_topmost_title(readme_text)
        write_project_file(out_dir, repo_name, project_link, title, readme_text)
        if delay_seconds > 0:
            time.sleep(delay_seconds)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    # Load .env from project root so env-backed defaults work for token/owner
    project_root = get_project_root()
    load_dotenv(project_root / ".env")

    # Default output directory: <project_root>/temp_files (overridable via OUT_DIR env or --out-dir)
    default_out_dir = os.environ.get("OUT_DIR") or str(project_root / "temp_files")

    parser = argparse.ArgumentParser(
        description=(
            "Fetch all public GitHub repositories for a user/org and write per-project files "
            "containing link, README, and topmost README title."
        )
    )
    parser.add_argument(
        "owner",
        nargs="?",
        default=os.environ.get("GH_OWNER", "Gouranga-GH"),
        help="GitHub username or organization (default from GH_OWNER env or 'Gouranga-GH')",
    )
    parser.add_argument(
        "--out-dir",
        default=default_out_dir,
        help="Output directory to write project files (default '<project_root>/temp_files')",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN"),
        help="GitHub token to increase rate limits (optional; can be set via GITHUB_TOKEN in .env)",
    )
    parser.add_argument("--include-forks", action="store_true", help="Include forked repositories as well")
    parser.add_argument("--delay", type=float, default=0.0, help="Optional delay in seconds between API calls to avoid rate limiting")
    return parser.parse_args(list(argv) if argv is not None else None)


def main() -> None:
    args = parse_args()
    try:
        fetch_and_write(owner=args.owner, out_dir=args.out_dir, token=args.token, include_forks=args.include_forks, delay_seconds=args.delay)
        print(f"Wrote project files to: {os.path.abspath(args.out_dir)}")
    except urllib.error.HTTPError as e:
        print(f"HTTP error {e.code}: {e.reason}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Network error: {e.reason}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
