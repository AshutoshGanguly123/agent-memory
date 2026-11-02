"""Download and hash pinned public test-only assets in the new experiment."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'memalpha': ('YuWangX/Memalpha-4B', '8c5683d0196a846ef1048b1fd0d4f9c7196e22d6'),
    'qwen3_base': ('Qwen/Qwen3-4B', '1cfa9a7208912126459214e8b04321603b3df60c'),
    'reader': ('Qwen/Qwen3-32B', '9216db5781bf21249d130ec9da846c4624c16137'),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as out:
        json.dump(value, out, indent=2, sort_keys=True)
        out.flush()
        os.fsync(out.fileno())
    tmp.replace(path)


def download(alias, repo, revision):
    from huggingface_hub import HfApi, snapshot_download
    target = ROOT / 'models' / alias
    status = ROOT / 'results/setup' / f'{alias}.download.json'
    state = dict(alias=alias, repo_id=repo, revision=revision, started_utc=now(), state='downloading')
    save(status, state)
    try:
        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        if info.sha != revision:
            raise RuntimeError(f'{alias}: model revision mismatch')
        snapshot_download(repo, revision=revision, local_dir=str(target), max_workers=4,
                          allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja',
                                          'LICENSE*', 'README.md'])
        state.update(state='verifying', downloaded_utc=now())
        save(status, state)
        files = {}
        expected = {x.rfilename: x for x in info.siblings}
        for path in sorted(target.rglob('*')):
            if not path.is_file() or '.cache' in path.relative_to(target).parts:
                continue
            relative = str(path.relative_to(target))
            digest = hashlib.sha256()
            with path.open('rb') as source:
                for chunk in iter(lambda: source.read(16 * 1024 * 1024), b''):
                    digest.update(chunk)
            sha = digest.hexdigest()
            remote = expected.get(relative)
            remote_sha = getattr(getattr(remote, 'lfs', None), 'sha256', None)
            if remote_sha and sha != remote_sha:
                raise RuntimeError(f'{alias}/{relative}: LFS SHA256 mismatch')
            files[relative] = {'size': path.stat().st_size, 'sha256': sha, 'remote_lfs_sha256': remote_sha}
        if not any(name.endswith('.safetensors') for name in files):
            raise RuntimeError(f'{alias}: no model weights downloaded')
        state.update(state='ready', completed_utc=now(), files=files)
        save(status, state)
        print(json.dumps({'alias': alias, 'state': 'ready', 'files': len(files)}), flush=True)
        return state
    except BaseException as exc:
        state.update(state='failed', finished_utc=now(), error=str(exc))
        save(status, state)
        raise


if __name__ == '__main__':
    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    status = ROOT / 'results/setup/assets.exit.json'
    results, errors = {}, []
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(download, alias, *args): alias for alias, args in MODELS.items()}
        for task in as_completed(pending):
            try:
                results[pending[task]] = task.result()
            except BaseException:
                errors.append(pending[task])
                traceback.print_exc()
    save(ROOT / 'models/manifest.json', {'created_utc': now(), 'models': results})
    save(status, {'exit_code': int(bool(errors)), 'failed_models': errors, 'finished_utc': now()})
    raise SystemExit(int(bool(errors)))
