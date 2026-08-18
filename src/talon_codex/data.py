from __future__ import annotations

import hashlib
import math
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, Sampler, get_worker_info

try:
    import albumentations as A
except ImportError:  # surfaced when a loader is actually built
    A = None

from .config import ResearchConfig


ROI_LABELS = ("tiny", "small", "medium", "large")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_data_worker(worker_id: int) -> None:
    worker_seed = int(torch.initial_seed() % (2**32))
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    worker = get_worker_info()
    if worker is not None and hasattr(worker.dataset, "rng"):
        worker.dataset.rng = np.random.default_rng(worker_seed)


def _ascii_slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_value.lower()).strip("_") or "subject"


def talon_mask_name(value: str) -> str:
    text = str(value).strip()
    if len(text) <= 2:
        return "*" * len(text)
    return text[0] + ("*" * (len(text) - 2)) + text[-1]


def _resolve_from_metadata(raw_path: str, root: Path, class_name: str, image_name: str, kind: str) -> Path | None:
    raw = Path(str(raw_path))
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.append(root / raw.name)
    subdir = "original" if kind == "image" else "segmentation"
    class_dir = root / class_name / subdir
    stem = Path(image_name).stem
    suffixes = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
    candidates.extend(class_dir / f"{stem}{suffix}" for suffix in suffixes)
    candidates.append(class_dir / raw.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def read_image_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    if image.max(initial=0.0) > 1.0:
        image /= 255.0
    return np.clip(image, 0.0, 1.0)


def read_binary_mask(path: str | Path, threshold: int = 127) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Unable to read mask: {path}")
    return (mask > int(threshold)).astype(np.uint8)


def pixel_sha256(path: str | Path, grayscale: bool = False) -> str:
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED
    array = cv2.imread(str(path), flag)
    if array is None:
        raise FileNotFoundError(path)
    header = f"{array.shape}|{array.dtype}".encode("utf-8")
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_metadata(config: ResearchConfig) -> pd.DataFrame:
    columns = config.section("columns")
    required = [columns[k] for k in ("patient", "image_name", "image_path", "mask_path", "class_name")]
    frame = pd.read_excel(config.metadata_path)
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Metadata columns missing: {missing}")
    spacing_config = config.get("physical_spacing_columns", {}) or {}
    optional_spacing_columns = [name for name in spacing_config.values() if name and name in frame.columns]
    frame = frame[required + [name for name in optional_spacing_columns if name not in required]].copy()
    for column in required:
        frame[column] = frame[column].astype(str).str.strip()
    frame[columns["class_name"]] = frame[columns["class_name"]].str.upper()

    frame["ResolvedImagePath"] = frame.apply(
        lambda row: _resolve_from_metadata(
            row[columns["image_path"]], config.dataset_root,
            row[columns["class_name"]], row[columns["image_name"]], "image"
        ), axis=1,
    )
    frame["ResolvedRoiPath"] = frame.apply(
        lambda row: _resolve_from_metadata(
            row[columns["mask_path"]], config.dataset_root,
            row[columns["class_name"]], row[columns["image_name"]], "mask"
        ), axis=1,
    )
    if frame["ResolvedImagePath"].isna().any() or frame["ResolvedRoiPath"].isna().any():
        bad = frame[frame["ResolvedImagePath"].isna() | frame["ResolvedRoiPath"].isna()]
        raise FileNotFoundError(f"Unresolved image/mask paths for {len(bad)} metadata rows.")

    class_to_id = {str(name).upper(): int(identifier) for name, identifier in config.get("class_mapping", {}).items()}
    unknown_classes = sorted(set(frame[columns["class_name"]]) - set(class_to_id))
    if unknown_classes:
        raise ValueError(f"Classes missing from class_mapping: {unknown_classes}")
    frame["class_id"] = frame[columns["class_name"]].map(class_to_id).astype(int)
    for standard_name, key in (("PixelSpacingXmm", "x_mm"), ("PixelSpacingYmm", "y_mm"), ("SliceThicknessMm", "slice_thickness_mm")):
        source_name = spacing_config.get(key)
        frame[standard_name] = pd.to_numeric(frame[source_name], errors="coerce") if source_name and source_name in frame else np.nan
    return add_anonymous_ids(frame, config)


def add_anonymous_ids(frame: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    patient_col = config.section("columns")["patient"]
    overrides = {str(k).strip(): str(v).strip() for k, v in config.get("subject_alias_overrides", {}).items()}
    result = frame.copy()
    result["SubjectGroupKey"] = result[patient_col].map(lambda value: overrides.get(str(value).strip(), str(value).strip()))
    unique_keys = sorted(result["SubjectGroupKey"].unique(), key=lambda value: _ascii_slug(str(value)))
    id_map = {key: f"patient{index:03d}" for index, key in enumerate(unique_keys, start=1)}
    result["SubjectID"] = result["SubjectGroupKey"].map(id_map)
    result["MaskedPatientName"] = result[patient_col].map(talon_mask_name)

    case_ids: dict[tuple[str, str], str] = {}
    case_counter: defaultdict[str, int] = defaultdict(int)
    for patient_name, subject_id in result[[patient_col, "SubjectID"]].drop_duplicates().itertuples(index=False):
        case_counter[subject_id] += 1
        case_ids[(subject_id, patient_name)] = f"{subject_id}_case{case_counter[subject_id]:02d}"
    result["CaseID"] = [case_ids[(subject, patient)] for subject, patient in zip(result["SubjectID"], result[patient_col])]
    return result


def audit_metadata(frame: pd.DataFrame, config: ResearchConfig, include_hashes: bool = True) -> dict[str, pd.DataFrame]:
    columns = config.section("columns")
    threshold = int(config.get("mask_threshold", 127))
    mask_rows: list[dict[str, Any]] = []
    hash_rows: list[dict[str, Any]] = []
    row_columns = [columns["patient"], columns["image_name"], columns["image_path"], columns["mask_path"], columns["class_name"]]
    duplicate_rows = frame[frame.duplicated(row_columns, keep=False)].sort_values(row_columns).copy()
    duplicate_paths = frame[frame.duplicated("ResolvedImagePath", keep=False)].sort_values("ResolvedImagePath").copy()
    for row in frame.itertuples(index=False):
        item = row._asdict()
        image_path = Path(item["ResolvedImagePath"])
        mask_path = Path(item["ResolvedRoiPath"])
        raw_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw_image is None:
            raise FileNotFoundError(image_path)
        if raw_mask is None:
            raise FileNotFoundError(mask_path)
        mask_gt0 = raw_mask > 0
        mask_clean = raw_mask > threshold
        mask_rows.append({
            "SubjectID": item["SubjectID"], "CaseID": item["CaseID"],
            "Split": item.get("Split", "unassigned"),
            "ImageName": item[columns["image_name"]], "MaskPath": str(mask_path),
            "MaskExtension": mask_path.suffix.lower(), "UniqueIntensityCount": int(np.unique(raw_mask).size),
            "AreaGt0Px": int(mask_gt0.sum()), "AreaGt127Px": int(mask_clean.sum()),
            "AreaInflationRatio": float(mask_gt0.sum() / max(mask_clean.sum(), 1)),
            "ImageHeight": int(raw_image.shape[0]), "ImageWidth": int(raw_image.shape[1]),
            "MaskHeight": int(raw_mask.shape[0]), "MaskWidth": int(raw_mask.shape[1]),
            "ImageMaskShapeMatch": bool(raw_image.shape[:2] == raw_mask.shape[:2]),
            "IsBinaryAfterThreshold": bool(np.isin(mask_clean.astype(np.uint8), [0, 1]).all()),
            "IsEmptyAfterThreshold": bool(mask_clean.sum() == 0),
        })
        if include_hashes:
            hash_rows.append({
                "SubjectID": item["SubjectID"], "CaseID": item["CaseID"],
                "Split": item.get("Split", "unassigned"),
                "ImageName": item[columns["image_name"]],
                "ResolvedImagePath": str(item["ResolvedImagePath"]), "ResolvedRoiPath": str(mask_path),
                "ImageFileSHA256": file_sha256(item["ResolvedImagePath"]),
                "MaskFileSHA256": file_sha256(mask_path),
                "ImagePixelSHA256": pixel_sha256(item["ResolvedImagePath"]),
                "MaskPixelSHA256": pixel_sha256(mask_path, grayscale=True),
            })
    mask_audit = pd.DataFrame(mask_rows)
    hashes = pd.DataFrame(hash_rows)
    duplicates = pd.DataFrame()
    duplicate_bytes = pd.DataFrame()
    if not hashes.empty:
        duplicate_bytes = hashes[hashes.duplicated("ImageFileSHA256", keep=False)].sort_values("ImageFileSHA256")
        duplicates = hashes[hashes.duplicated("ImagePixelSHA256", keep=False)].sort_values("ImagePixelSHA256")
        if not duplicates.empty:
            risks = {}
            for digest, group in duplicates.groupby("ImagePixelSHA256"):
                if group["Split"].nunique() > 1:
                    risks[digest] = "cross_split_leakage"
                elif group["SubjectID"].nunique() > 1:
                    risks[digest] = "cross_subject_duplicate"
                else:
                    risks[digest] = "within_subject_repeat"
            duplicates["DuplicateRisk"] = duplicates["ImagePixelSHA256"].map(risks)
    return {
        "mask_audit": mask_audit, "pixel_hashes": hashes, "duplicate_images": duplicates,
        "duplicate_rows": duplicate_rows, "duplicate_paths": duplicate_paths,
        "duplicate_file_bytes": duplicate_bytes,
    }


def validate_expected_cohort(frame: pd.DataFrame, config: ResearchConfig) -> None:
    expected = config.section("expected_clean_cohort")
    columns = config.section("columns")
    observed_records = int(frame[columns["patient"]].nunique())
    observed_subjects = int(frame["SubjectID"].nunique())
    observed_cases = int(frame["CaseID"].nunique())
    observed_slices = int(len(frame))
    class_slices = frame[columns["class_name"]].value_counts().to_dict()
    class_cases = frame.groupby(columns["class_name"])["CaseID"].nunique().to_dict()
    class_subjects = frame.groupby(columns["class_name"])["SubjectID"].nunique().to_dict()
    errors: list[str] = []
    if observed_records != int(expected["metadata_case_records"]):
        errors.append(f"metadata_case_records {observed_records} != {expected['metadata_case_records']}")
    if observed_subjects != int(expected["subjects"]):
        errors.append(f"subjects {observed_subjects} != {expected['subjects']}")
    if observed_cases != int(expected["cases"]):
        errors.append(f"cases {observed_cases} != {expected['cases']}")
    if observed_slices != int(expected["slices"]):
        errors.append(f"slices {observed_slices} != {expected['slices']}")
    if class_slices != expected["class_slices"]:
        errors.append(f"class_slices {class_slices} != {expected['class_slices']}")
    if class_cases != expected["class_cases"]:
        errors.append(f"class_cases {class_cases} != {expected['class_cases']}")
    if class_subjects != expected["class_subjects"]:
        errors.append(f"class_subjects {class_subjects} != {expected['class_subjects']}")
    identity_assertions = config.section("metadata_identity_assertions")
    patient_names = set(frame[columns["patient"]].astype(str).str.strip())
    missing_required = sorted(set(identity_assertions.get("required_patient_names", [])) - patient_names)
    present_forbidden = sorted(set(identity_assertions.get("forbidden_patient_names", [])) & patient_names)
    if missing_required:
        errors.append(f"required patient records absent: {missing_required}")
    if present_forbidden:
        errors.append(f"forbidden superseded patient records present: {present_forbidden}")
    aliases = pd.DataFrame(
        [(str(name).strip(), str(group).strip()) for name, group in config.get("subject_alias_overrides", {}).items()],
        columns=["PatientNameExpected", "AliasGroup"],
    )
    if not aliases.empty:
        for alias_group, expected_names in aliases.groupby("AliasGroup"):
            names = set(expected_names["PatientNameExpected"])
            linked = frame[frame[columns["patient"]].isin(names)]
            found_names = set(linked[columns["patient"]])
            if found_names != names:
                errors.append(f"alias group {alias_group} names {sorted(found_names)} != {sorted(names)}")
            if linked["SubjectID"].nunique() != 1:
                errors.append(f"alias group {alias_group} does not map to exactly one SubjectID")
            if linked["CaseID"].nunique() != len(names):
                errors.append(f"alias group {alias_group} must retain {len(names)} separate CaseID values")
    if errors:
        raise ValueError("Clean cohort validation failed: " + "; ".join(errors))


def export_cleaned_binary_masks(frame: pd.DataFrame, config: ResearchConfig, output_root: Path) -> pd.DataFrame:
    """Write thresholded PNG copies without modifying any source mask."""
    records = []
    for row in frame.itertuples(index=False):
        item = row._asdict()
        source = Path(item["ResolvedRoiPath"])
        raw = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(source)
        cleaned = (raw > int(config.get("mask_threshold", 127))).astype(np.uint8)
        destination = output_root / str(item["ClassName"]) / str(item["SubjectID"]) / str(item["CaseID"]) / f"{Path(str(item['ImageName'])).stem}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), cleaned * 255):
            raise OSError(f"Could not write cleaned mask: {destination}")
        records.append({
            "SubjectID": item["SubjectID"], "CaseID": item["CaseID"], "ImageName": item["ImageName"],
            "source_mask": str(source), "cleaned_mask": str(destination),
            "area_gt0_px": int((raw > 0).sum()), "area_gt127_px": int(cleaned.sum()),
            "area_removed_px": int((raw > 0).sum() - cleaned.sum()), "is_binary_0_255": True,
        })
    manifest = pd.DataFrame(records)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_root / "cleaned_mask_manifest.csv", index=False)
    return manifest


def grouped_patient_split(frame: pd.DataFrame, config: ResearchConfig, seed: int | None = None) -> dict[str, pd.DataFrame]:
    seed = int(config.get("split_seed", config.seed) if seed is None else seed)
    split_cfg = config.section("split")
    class_col = config.section("columns")["class_name"]
    summary = frame.groupby("SubjectID", as_index=False).agg(
        class_id=("class_id", "first"), ClassName=(class_col, "first"), n_slices=("CaseID", "size")
    )
    if frame.groupby("SubjectID")["class_id"].nunique().max() > 1:
        raise ValueError("A SubjectID maps to more than one class.")
    trainval, test = train_test_split(
        summary, test_size=float(split_cfg["test_fraction"]), random_state=seed, stratify=summary["class_id"]
    )
    relative_val = float(split_cfg["validation_fraction"]) / (1.0 - float(split_cfg["test_fraction"]))
    train, validation = train_test_split(
        trainval, test_size=relative_val, random_state=seed, stratify=trainval["class_id"]
    )
    assignments = {
        "train": set(train["SubjectID"]),
        "validation": set(validation["SubjectID"]),
        "test": set(test["SubjectID"]),
    }
    assert assignments["train"].isdisjoint(assignments["validation"])
    assert assignments["train"].isdisjoint(assignments["test"])
    assert assignments["validation"].isdisjoint(assignments["test"])
    result = {
        name: frame[frame["SubjectID"].isin(subjects)].copy().reset_index(drop=True)
        for name, subjects in assignments.items()
    }
    for name, split_frame in result.items():
        split_frame["Split"] = name
    return result


def fit_train_roi_bins(train_frame: pd.DataFrame, config: ResearchConfig) -> tuple[pd.DataFrame, list[float]]:
    threshold = int(config.get("mask_threshold", 127))
    result = train_frame.copy()
    result["RoiAreaPxNative"] = [
        int(read_binary_mask(path, threshold).sum()) for path in result["ResolvedRoiPath"]
    ]
    quantiles = result["RoiAreaPxNative"].quantile([0.25, 0.50, 0.75]).to_numpy(dtype=float)
    edges = [-math.inf] + sorted(set(float(value) for value in quantiles)) + [math.inf]
    if len(edges) != 5:
        ranked = result["RoiAreaPxNative"].rank(method="first")
        result["RoiSizeBin"] = pd.qcut(ranked, q=4, labels=ROI_LABELS)
        fallback = result.groupby("RoiSizeBin", observed=True)["RoiAreaPxNative"].max().iloc[:-1].tolist()
        edges = [-math.inf] + [float(value) for value in fallback] + [math.inf]
    else:
        result["RoiSizeBin"] = pd.cut(result["RoiAreaPxNative"], bins=edges, labels=ROI_LABELS, include_lowest=True)
    return result, edges


def apply_train_roi_bins(frame: pd.DataFrame, edges: Sequence[float], config: ResearchConfig) -> pd.DataFrame:
    threshold = int(config.get("mask_threshold", 127))
    result = frame.copy()
    result["RoiAreaPxNative"] = [int(read_binary_mask(path, threshold).sum()) for path in result["ResolvedRoiPath"]]
    result["RoiSizeBin"] = pd.cut(result["RoiAreaPxNative"], bins=list(edges), labels=ROI_LABELS, include_lowest=True)
    return result


def build_grouped_cv(frame: pd.DataFrame, config: ResearchConfig) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    cv = config.section("cross_validation")
    splitter = StratifiedGroupKFold(n_splits=int(cv["folds"]), shuffle=True, random_state=int(config.get("split_seed", config.seed)))
    yield from splitter.split(frame, y=frame["class_id"], groups=frame["SubjectID"])


def _tiny_crop(image: np.ndarray, mask: np.ndarray, cfg: dict[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if mask.sum() > int(cfg["tiny_area_max_px_native"]) or rng.random() >= float(cfg["tiny_crop_probability"]):
        return image, mask
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return image, mask
    margin = int(cfg["tiny_crop_margin_px"])
    min_side = int(cfg["tiny_crop_min_side_px"])
    x0, x1 = max(0, xs.min() - margin), min(mask.shape[1], xs.max() + margin + 1)
    y0, y1 = max(0, ys.min() - margin), min(mask.shape[0], ys.max() + margin + 1)
    side = max(x1 - x0, y1 - y0, min_side)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    x0 = max(0, min(mask.shape[1] - side, cx - side // 2))
    y0 = max(0, min(mask.shape[0] - side, cy - side // 2))
    return image[y0:y0 + side, x0:x0 + side], mask[y0:y0 + side, x0:x0 + side]


def _build_transform(config: ResearchConfig, training: bool):
    if A is None:
        raise ImportError("albumentations is required to build datasets.")
    size = int(config.get("image_size", 512))
    aug = config.section("augmentation")
    if not training or not bool(aug["enabled"]):
        return A.Compose([A.Resize(size, size)])
    return A.Compose([
        A.HorizontalFlip(p=float(aug["horizontal_flip_probability"])),
        A.ShiftScaleRotate(
            shift_limit=float(aug["shift_limit"]), scale_limit=float(aug["scale_limit"]),
            rotate_limit=int(aug["rotate_limit_degrees"]), border_mode=cv2.BORDER_CONSTANT,
            value=0, mask_value=0, p=float(aug["geometric_probability"]),
        ),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=float(aug["brightness_contrast_limit"]),
                contrast_limit=float(aug["brightness_contrast_limit"]), p=1.0,
            ),
            A.GaussNoise(var_limit=tuple(float(v * 255 * 255) for v in aug["noise_variance"]), p=1.0),
        ], p=float(aug["intensity_probability"])),
        A.Resize(size, size),
    ])


class TalonDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, config: ResearchConfig, training: bool = False, seed: int | None = None):
        self.frame = frame.reset_index(drop=True)
        self.config = config
        self.training = training
        self.transform = _build_transform(config, training)
        self.rng = np.random.default_rng(config.seed if seed is None else seed)
        self.threshold = int(config.get("mask_threshold", 127))

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image = read_image_rgb(row["ResolvedImagePath"])
        mask = read_binary_mask(row["ResolvedRoiPath"], self.threshold)
        if self.training:
            image, mask = _tiny_crop(image, mask, self.config.section("augmentation"), self.rng)
        transformed = self.transform(image=image, mask=mask)
        aug_image = transformed["image"].astype(np.float32)
        aug_mask = (transformed["mask"] > 0).astype(np.float32)
        if self.training and aug_mask.sum() == 0 and mask.sum() > 0:
            fallback = _build_transform(self.config, False)(image=image, mask=mask)
            aug_image = fallback["image"].astype(np.float32)
            aug_mask = (fallback["mask"] > 0).astype(np.float32)
        return {
            "image": torch.from_numpy(np.transpose(aug_image, (2, 0, 1))).float(),
            "mask": torch.from_numpy(aug_mask[None]).float(),
            "class_id": torch.tensor(int(row["class_id"]), dtype=torch.long),
            "SubjectID": row["SubjectID"], "CaseID": row["CaseID"],
            "MaskedPatientName": row["MaskedPatientName"], "ImageName": row[self.config.section("columns")["image_name"]],
            "ClassName": row[self.config.section("columns")["class_name"]], "RoiSizeBin": str(row.get("RoiSizeBin", "unknown")),
            "ResolvedImagePath": str(row["ResolvedImagePath"]), "ResolvedRoiPath": str(row["ResolvedRoiPath"]),
            "PixelSpacingXmm": float(row.get("PixelSpacingXmm", np.nan)),
            "PixelSpacingYmm": float(row.get("PixelSpacingYmm", np.nan)),
            "SliceThicknessMm": float(row.get("SliceThicknessMm", np.nan)),
        }


class BalancedBatchSampler(Sampler[list[int]]):
    def __init__(self, frame: pd.DataFrame, batch_size: int, seed: int):
        if batch_size < 2:
            raise ValueError("Balanced batches require batch_size >= 2.")
        self.frame = frame.reset_index(drop=True)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.pools: dict[tuple[int, str], np.ndarray] = {}
        for (class_id, size_bin), group in self.frame.groupby(["class_id", "RoiSizeBin"], observed=True):
            self.pools[(int(class_id), str(size_bin))] = group.index.to_numpy(dtype=int)
        self.classes = sorted(self.frame["class_id"].unique().astype(int).tolist())
        self.last_epoch_indices: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return math.ceil(len(self.frame) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.last_epoch_indices = []
        available = {key: value for key, value in self.pools.items() if len(value)}
        for _ in range(len(self)):
            batch: list[int] = []
            for class_id in self.classes:
                keys = [key for key in available if key[0] == class_id]
                if keys:
                    key = keys[int(rng.integers(0, len(keys)))]
                    batch.append(int(rng.choice(available[key])))
            all_keys = list(available)
            while len(batch) < self.batch_size:
                key = all_keys[int(rng.integers(0, len(all_keys)))]
                batch.append(int(rng.choice(available[key])))
            rng.shuffle(batch)
            self.last_epoch_indices.extend(batch)
            yield batch

    def exposure_summary(self) -> pd.DataFrame:
        if not self.last_epoch_indices:
            return pd.DataFrame(columns=["epoch", "SubjectID", "class_id", "RoiSizeBin", "exposures"])
        sampled = self.frame.iloc[self.last_epoch_indices]
        summary = sampled.groupby(["SubjectID", "class_id", "RoiSizeBin"], observed=True).size().rename("exposures").reset_index()
        summary.insert(0, "epoch", self.epoch)
        return summary


@dataclass
class DataBundle:
    frames: dict[str, pd.DataFrame]
    loaders: dict[str, DataLoader]
    roi_edges: list[float]


def build_data_bundle(config: ResearchConfig, seed: int | None = None) -> DataBundle:
    frame = load_metadata(config)
    validate_expected_cohort(frame, config)
    splits = grouped_patient_split(frame, config, int(config.get("split_seed", config.seed)))
    train, edges = fit_train_roi_bins(splits["train"], config)
    validation = apply_train_roi_bins(splits["validation"], edges, config)
    test = apply_train_roi_bins(splits["test"], edges, config)
    frames = {"train": train, "validation": validation, "test": test}
    loader_cfg = config.section("loader")
    run_seed = config.seed if seed is None else int(seed)
    generator = torch.Generator().manual_seed(run_seed)
    datasets = {
        name: TalonDataset(split_frame, config, training=(name == "train"), seed=seed)
        for name, split_frame in frames.items()
    }
    if bool(loader_cfg["balanced_sampler"]):
        sampler = BalancedBatchSampler(train, int(loader_cfg["batch_size"]), config.seed if seed is None else seed)
        train_loader = DataLoader(
            datasets["train"], batch_sampler=sampler, num_workers=int(loader_cfg["num_workers"]),
            pin_memory=bool(loader_cfg["pin_memory"]), worker_init_fn=seed_data_worker, generator=generator,
        )
    else:
        train_loader = DataLoader(
            datasets["train"], batch_size=int(loader_cfg["batch_size"]), shuffle=True,
            num_workers=int(loader_cfg["num_workers"]), pin_memory=bool(loader_cfg["pin_memory"]),
            worker_init_fn=seed_data_worker, generator=generator,
        )
    loaders = {"train": train_loader}
    for name in ("validation", "test"):
        loaders[name] = DataLoader(
            datasets[name], batch_size=int(loader_cfg["batch_size"]), shuffle=False,
            num_workers=int(loader_cfg["num_workers"]), pin_memory=bool(loader_cfg["pin_memory"]),
            worker_init_fn=seed_data_worker, generator=generator,
        )
    return DataBundle(frames=frames, loaders=loaders, roi_edges=edges)


def build_cross_validation_bundle(
    frame: pd.DataFrame,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    config: ResearchConfig,
    seed: int,
) -> DataBundle:
    """Build an actual grouped-CV development fold without touching the locked test split."""
    train, edges = fit_train_roi_bins(frame.iloc[list(train_indices)].copy().reset_index(drop=True), config)
    validation = apply_train_roi_bins(frame.iloc[list(validation_indices)].copy().reset_index(drop=True), edges, config)
    frames = {"train": train, "validation": validation, "test": validation.copy()}
    loader_cfg = config.section("loader")
    generator = torch.Generator().manual_seed(int(seed))
    datasets = {name: TalonDataset(part, config, training=(name == "train"), seed=seed) for name, part in frames.items()}
    sampler = BalancedBatchSampler(train, int(loader_cfg["batch_size"]), int(seed))
    loaders = {
        "train": DataLoader(datasets["train"], batch_sampler=sampler, num_workers=int(loader_cfg["num_workers"]), pin_memory=bool(loader_cfg["pin_memory"]), worker_init_fn=seed_data_worker, generator=generator),
        "validation": DataLoader(datasets["validation"], batch_size=int(loader_cfg["batch_size"]), shuffle=False, num_workers=int(loader_cfg["num_workers"]), pin_memory=bool(loader_cfg["pin_memory"]), worker_init_fn=seed_data_worker, generator=generator),
    }
    loaders["test"] = loaders["validation"]
    return DataBundle(frames=frames, loaders=loaders, roi_edges=edges)


def build_external_loader(metadata_path: str | Path, config: ResearchConfig, roi_edges: Sequence[float]) -> tuple[pd.DataFrame, DataLoader]:
    """Load an independent cohort with the same schema; no cohort-count assertion."""
    from .config import ResearchConfig
    external_raw = dict(config.raw)
    external_raw["metadata_path"] = str(Path(metadata_path).resolve())
    external_raw["dataset_root"] = str(Path(metadata_path).resolve().parent)
    external_config = ResearchConfig(external_raw, Path(metadata_path).resolve())
    frame = load_metadata(external_config)
    frame = apply_train_roi_bins(frame, roi_edges, external_config)
    dataset = TalonDataset(frame, external_config, training=False, seed=config.seed)
    loader_cfg = config.section("loader")
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(dataset, batch_size=int(loader_cfg["batch_size"]), shuffle=False, num_workers=int(loader_cfg["num_workers"]), pin_memory=bool(loader_cfg["pin_memory"]), worker_init_fn=seed_data_worker, generator=generator)
    return frame, loader
