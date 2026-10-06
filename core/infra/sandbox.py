"""project sandbox roots — path ที่โค้ดอนุญาตให้แตะได้ (Fix.md H1)

worktree ใช้ `data_cache/` (และบางที `exports/`) ร่วมกับ main repo ผ่าน symlink —
`resolve()` จะข้ามไป main repo ได้ ดังนั้น roots ต้องรวม "target จริง" ของ
โฟลเดอร์ที่ project ชี้出去 ด้วย ไม่งั้น worktree ถูกบล็อกเอง (เทสต์ตก)
"""

from __future__ import annotations

from pathlib import Path

# โฟลเดอร์ข้อมูล project ที่อนุญาตเสมอ (อาจเป็น symlink → resolve target ด้วย)
_SHARED_DIRS = ("data_cache", "exports")


def project_roots(repo: Path, *extra: str | Path) -> tuple[Path, ...]:
    """roots สำหรับ `is_relative_to`: repo + `_SHARED_DIRS` (resolve แล้ว) + `extra`

    `repo` ให้ caller ส่งมา (module-level REPO_ROOT — patch ได้ในเทสต์);
    `extra` = โฟลเดอร์ config ที่ patch ได้ เช่น MODELS_DIR / DATASETS_DIR
    """
    roots = [Path(repo).resolve()]
    for name in _SHARED_DIRS:
        base = Path(repo) / name
        if base.exists():
            roots.append(base.resolve())
    roots.extend(Path(value).resolve() for value in extra)
    return tuple(roots)
