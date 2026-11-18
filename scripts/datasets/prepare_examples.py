#!/usr/bin/env python3
"""Prepara e importa ejemplos Imagenette/COCO sin Python en el anfitrión.

Se ejecuta en el contenedor backend mediante los lanzadores .sh o .ps1.
Las descargas quedan en /datasets, un volumen local ignorado por Git.
"""
from __future__ import annotations

import argparse
import csv
import getpass
import json
import mimetypes
import os
import random
import shutil
import sys
import tarfile
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path, PurePosixPath

import requests
from PIL import Image

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
COCO_BUCKET = "https://s3.amazonaws.com/images.cocodataset.org"
COCO_ARCHIVE_URL = f"{COCO_BUCKET}/annotations/annotations_trainval2017.zip"
COCO_MEMBER = "annotations/instances_val2017.json"
IMAGENETTE_LABELS = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}
RETRY_STATUSES = {429, 500, 502, 503, 504}
DOWNLOAD_ATTEMPTS = 6
API_ATTEMPTS = 4
BATCH_SIZE = 10


def validate_count(count: int) -> int:
    if count < 10 or count > 100 or count % 10:
        raise ValueError("--count debe ser un múltiplo de 10 entre 10 y 100")
    return count


def pause(attempt: int) -> None:
    # Backoff exponencial acotado: no dormir en el último intento.
    time.sleep(min(2 ** (attempt - 1), 12))


def download(url: str, destination: Path, *, attempts: int = DOWNLOAD_ATTEMPTS) -> None:
    """Descarga reanudable por HTTP Range; conserva .part si hay un corte.

    Para respuestas que ignoren Range, reinicia la descarga en lugar de añadir
    bytes a un archivo incompatible. El llamador valida el contenido descargado.
    """
    if destination.is_file() and destination.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    for attempt in range(1, attempts + 1):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "TFG-Example-Datasets/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(20, 90)) as response:
                if response.status_code == 416:
                    part.unlink(missing_ok=True)
                    raise requests.RequestException("El servidor rechazó el rango de reanudación")
                response.raise_for_status()
                append = bool(offset and response.status_code == 206)
                if append:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        part.unlink(missing_ok=True)
                        raise requests.RequestException("Rango HTTP de reanudación inesperado")
                with part.open("ab" if append else "wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
                # Si hubo truncamiento silencioso y Content-Length era conocido,
                # requests/urllib3 normalmente lo detecta. Comprobamos también
                # nosotros el tamaño escrito en esta respuesta.
                advertised = response.headers.get("Content-Length")
                if advertised is not None:
                    expected = int(advertised)
                    actual = part.stat().st_size - (offset if append else 0)
                    if actual != expected:
                        raise requests.RequestException("Descarga HTTP incompleta")
                part.replace(destination)
                return
        except (requests.RequestException, OSError, ValueError) as error:
            if isinstance(error, requests.HTTPError) and error.response is not None:
                if error.response.status_code not in RETRY_STATUSES:
                    raise RuntimeError(f"No se puede descargar {url}: HTTP {error.response.status_code}") from error
            if attempt >= attempts:
                raise RuntimeError(
                    f"No se pudo descargar {url} tras {attempts} intentos. "
                    "Vuelve a ejecutar el mismo comando para continuar."
                ) from error
            print(f"Aviso: descarga interrumpida ({attempt}/{attempts}); reintentando...", flush=True)
            pause(attempt)


def ensure_archive(url: str, target: Path, validator) -> None:
    """Reintenta una descarga aparentemente completa pero corrupta."""
    for attempt in range(1, 4):
        if not target.exists():
            print(f"Descargando {target.name} (se reutiliza en ejecuciones posteriores)...", flush=True)
            download(url, target)
        try:
            validator(target)
            return
        except (OSError, EOFError, ValueError, tarfile.TarError, zipfile.BadZipFile) as error:
            target.unlink(missing_ok=True)
            target.with_name(target.name + ".part").unlink(missing_ok=True)
            if attempt == 3:
                raise RuntimeError(f"El archivo {target.name} está dañado") from error
            print(f"Aviso: archivo {target.name} incompleto o inválido; nueva descarga.", flush=True)


def validate_imagenette_archive(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        if not any(
            member.isfile() and PurePosixPath(member.name).parts[:2] == ("imagenette2-320", "val")
            for member in archive.getmembers()
        ):
            raise ValueError("No se encontraron imágenes de validación Imagenette")


def valid_image(path: Path, *, width: int | None = None, height: int | None = None) -> bool:
    try:
        with Image.open(path) as image:
            if width is not None and image.width != width:
                return False
            if height is not None and image.height != height:
                return False
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def atomic_copy(source, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".download-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            shutil.copyfileobj(source, output)
        os.replace(temp_name, target)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def prepare_imagenette(base: Path, count: int) -> tuple[Path, list[dict[str, str]]]:
    archive_path = base / ".cache" / "imagenette2-320.tgz"
    ensure_archive(IMAGENETTE_URL, archive_path, validate_imagenette_archive)
    per_class = count // 10
    result_dir = base / "classification" / f"imagenette_{count}"
    sample_dir = result_dir / "images"
    rows = []
    # Extraemos únicamente el split de validación; nunca enlaces, rutas
    # absolutas ni componentes ajenos al patrón conocido del archivo.
    with tarfile.open(archive_path, "r:gz") as archive:
        by_synset: dict[str, list[tarfile.TarInfo]] = defaultdict(list)
        for member in archive.getmembers():
            parts = PurePosixPath(member.name).parts
            if (
                member.isfile()
                and len(parts) == 4
                and parts[:2] == ("imagenette2-320", "val")
                and parts[2] in IMAGENETTE_LABELS
                and parts[3].lower().endswith((".jpg", ".jpeg", ".png"))
            ):
                by_synset[parts[2]].append(member)
        rng = random.Random(42)
        for synset, label in IMAGENETTE_LABELS.items():
            candidates = sorted(by_synset[synset], key=lambda item: item.name)
            if len(candidates) < per_class:
                raise RuntimeError(f"Imagenette: no hay suficientes imágenes de {synset}")
            selected = rng.sample(candidates, per_class)
            for position, member in enumerate(selected, start=1):
                original = PurePosixPath(member.name).name
                filename = f"{synset}_{position:02d}_{original}"
                target = sample_dir / filename
                if not target.exists() or not valid_image(target):
                    source = archive.extractfile(member)
                    if source is None:
                        raise RuntimeError(f"No se pudo extraer {member.name}")
                    with source:
                        atomic_copy(source, target)
                    if not valid_image(target):
                        target.unlink(missing_ok=True)
                        raise RuntimeError(f"Imagenette: imagen inválida {original}")
                rows.append({"filename": filename, "label": label, "synset": synset})
    rows.sort(key=lambda row: (row["synset"], row["filename"]))
    result_dir.mkdir(parents=True, exist_ok=True)
    manifest = result_dir / "labels.csv"
    with manifest.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["filename", "label", "synset"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Clasificación: {len(rows)} imágenes etiquetadas en {result_dir}")
    return result_dir, rows


def validate_coco_archive(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        if COCO_MEMBER not in archive.namelist():
            raise ValueError("No se encontraron las anotaciones COCO 2017")
        # Verificamos completamente el miembro que se utilizará.
        with archive.open(COCO_MEMBER) as handle:
            json.load(handle)


def select_coco_ids(document: dict, count: int, seed: int = 2026) -> list[int]:
    """Misma selección diversa y determinista que los scripts originales."""
    image_ids = {image["id"] for image in document["images"]}
    by_image: dict[int, set[int]] = defaultdict(set)
    frequencies: Counter[int] = Counter()
    for annotation in document["annotations"]:
        image_id, category_id = annotation.get("image_id"), annotation.get("category_id")
        if image_id not in image_ids or not isinstance(category_id, int):
            continue
        if category_id not in by_image[image_id]:
            by_image[image_id].add(category_id)
            frequencies[category_id] += 1
    candidates = sorted(image_id for image_id in image_ids if by_image[image_id])
    if len(candidates) < count:
        raise ValueError(f"COCO solo contiene {len(candidates)} imágenes anotadas")
    rng = random.Random(seed)
    tie = {image_id: rng.random() for image_id in candidates}
    chosen: list[int] = []
    chosen_set: set[int] = set()
    counts: Counter[int] = Counter()
    for _ in range(count):
        best = max(
            (image_id for image_id in candidates if image_id not in chosen_set),
            key=lambda image_id: (
                sum(counts[category] == 0 for category in by_image[image_id]),
                sum(1 / ((counts[category] + 1) * frequencies[category]) for category in by_image[image_id]),
                len(by_image[image_id]),
                tie[image_id],
                -image_id,
            ),
        )
        chosen.append(best)
        chosen_set.add(best)
        counts.update(by_image[best])
    return sorted(chosen)


def prepare_coco(base: Path, count: int) -> tuple[Path, dict]:
    archive_path = base / ".cache" / "annotations_trainval2017.zip"
    ensure_archive(COCO_ARCHIVE_URL, archive_path, validate_coco_archive)
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(COCO_MEMBER) as source:
            document = json.load(source)
    if not all(isinstance(document.get(field), list) for field in ("images", "annotations", "categories")):
        raise ValueError("Las anotaciones COCO originales no tienen el formato esperado")
    ids = set(select_coco_ids(document, count))
    images = sorted((image for image in document["images"] if image["id"] in ids), key=lambda image: image["id"])
    annotations = sorted((item for item in document["annotations"] if item.get("image_id") in ids), key=lambda a: a["id"])
    subset = {
        "info": {**document.get("info", {}), "description": "TFG: subconjunto COCO val2017 reproducible", "subset_image_count": count},
        "licenses": document.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": sorted(document["categories"], key=lambda category: category["id"]),
    }
    result_dir = base / "detection" / f"coco_{count}"
    image_dir = result_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    def fetch_image(image: dict) -> None:
        filename = PurePosixPath(image["file_name"]).name
        if filename != image["file_name"] or not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            raise ValueError("COCO contiene un nombre de imagen inesperado")
        target = image_dir / filename
        dimensions = {"width": image.get("width"), "height": image.get("height")}
        if valid_image(target, **dimensions):
            return
        target.unlink(missing_ok=True)
        url = f"{COCO_BUCKET}/val2017/{filename}"
        for attempt in range(1, 4):
            download(url, target)
            if valid_image(target, **dimensions):
                return
            target.unlink(missing_ok=True)
            if attempt == 3:
                raise RuntimeError(f"COCO: imagen dañada: {filename}")
        raise RuntimeError(f"COCO: imagen no válida {filename}")

    # Si un intento se corta, la próxima ejecución conserva las imágenes
    # verificadas y descarga únicamente las que faltan.
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(fetch_image, item): item for item in images}
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index % 10 == 0 or index == len(images):
                print(f"Detección: {index}/{len(images)} imágenes verificadas", flush=True)
    snapshot = result_dir / "instances_subset.json"
    temp = snapshot.with_suffix(".json.tmp")
    with temp.open("w", encoding="utf-8") as output:
        json.dump(subset, output, ensure_ascii=False, separators=(",", ":"))
    temp.replace(snapshot)
    print(f"Detección: {len(images)} imágenes y {len(annotations)} anotaciones en {result_dir}")
    return result_dir, subset


class ApiClient:
    def __init__(self, base: str, *, email: str | None = None) -> None:
        self.base = base.rstrip("/")
        self.session = requests.Session()
        token = os.environ.get("TFG_API_TOKEN", "").strip()
        if not token:
            if not sys.stdin.isatty():
                raise RuntimeError("Para importar sin terminal, define TFG_API_TOKEN o utiliza --prepare-only")
            email = email or input("Correo de tu cuenta de la plataforma: ").strip()
            password = getpass.getpass("Contraseña (no se mostrará): ")
            try:
                response = self.request("POST", "/auth/login", json={"email": email, "password": password})
                token = response.json()["access_token"]
            finally:
                del password
        self.session.headers.update({"Authorization": f"Bearer {token}"})
        me = self.request("GET", "/auth/me").json()
        self.user_id = me["user"]["id"]
        print(f"Sesión iniciada como {me['user']['email']}")

    def request(self, method: str, path: str, **kwargs) -> requests.Response:
        for attempt in range(1, API_ATTEMPTS + 1):
            try:
                response = self.session.request(method, self.base + path, timeout=(15, 120), **kwargs)
                response.raise_for_status()
                return response
            except requests.RequestException as error:
                status = getattr(getattr(error, "response", None), "status_code", None)
                if status is not None and status not in RETRY_STATUSES:
                    raise RuntimeError(f"La API rechazó {method} {path} (HTTP {status}): {str(error)[:250]}") from error
                if attempt == API_ATTEMPTS:
                    raise RuntimeError(f"Falló {method} {path} después de {API_ATTEMPTS} intentos") from error
                print(f"Aviso: API no disponible ({attempt}/{API_ATTEMPTS}); reintentando...", flush=True)
                pause(attempt)
        raise AssertionError("Bucle de reintentos incompleto")

    def list_datasets(self) -> list[dict]:
        return self.request("GET", "/datasets").json()

    def get_dataset(self, dataset_id: str) -> dict:
        return self.request("GET", f"/datasets/{dataset_id}").json()


def upload_batch(client: ApiClient, path: str, image_dir: Path, names: list[str], *, creation_name: str | None = None, task: str | None = None):
    with ExitStack() as stack:
        files = []
        for name in names:
            handle = stack.enter_context((image_dir / name).open("rb"))
            files.append(("files", (name, handle, mimetypes.guess_type(name)[0] or "image/jpeg")))
        payload = {"name": creation_name, "task_type": task} if creation_name is not None else None
        # No usamos reintentos ciegos para POST: si la conexión se corta,
        # primero reconciliamos el estado con la API para evitar duplicados.
        response = client.session.post(client.base + path, data=payload, files=files, timeout=(15, 180))
        response.raise_for_status()
        return response.json()


def names_in(dataset: dict) -> list[str]:
    return [item["filename"] for item in dataset.get("images", [])]


def get_known_dataset(client: ApiClient, state_path: Path, name: str, task: str, expected: set[str]) -> dict | None:
    # El estado incluye únicamente IDs, nunca JWT ni contraseñas.
    saved = {}
    if state_path.exists():
        try:
            saved = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            print("Aviso: estado de importación local ilegible; consultando la API.")
    ids = saved.get("datasets", {}) if isinstance(saved, dict) else {}
    if not isinstance(ids, dict):
        ids = {}
    remembered = ids.get(client.user_id)
    owned = client.list_datasets()
    matches = [dataset for dataset in owned if dataset["name"] == name and dataset.get("task_type") == task]
    if remembered:
        matches = [dataset for dataset in owned if dataset["id"] == remembered]
        if not matches:
            print("Aviso: el dataset asociado al estado local ya no existe; se buscará por nombre.")
            matches = [dataset for dataset in owned if dataset["name"] == name and dataset.get("task_type") == task]
    if len(matches) > 1:
        raise RuntimeError(f"Hay varios datasets llamados {name!r}. Renombra los duplicados antes de continuar.")
    dataset = matches[0] if matches else None
    if dataset:
        if dataset["name"] != name or dataset.get("task_type") != task:
            raise RuntimeError("El dataset guardado en el estado local cambió; no se modificará")
        if len(names_in(dataset)) != len(set(names_in(dataset))) or not set(names_in(dataset)).issubset(expected):
            raise RuntimeError("El dataset existente contiene imágenes ajenas o duplicadas; no se modificará")
        write_import_state(state_path, ids | {client.user_id: dataset["id"]})
    return dataset


def write_import_state(path: Path, identifiers: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps({"datasets": identifiers}, indent=2), encoding="utf-8")
    temp.replace(path)


def import_images(client: ApiClient, result_dir: Path, image_names: list[str], name: str, task: str) -> dict:
    image_dir = result_dir / "images"
    expected = set(image_names)
    if len(expected) != len(image_names):
        raise ValueError("Los nombres de imágenes del ejemplo deben ser únicos")
    state_path = result_dir / "import_state.json"
    dataset = get_known_dataset(client, state_path, name, task, expected)
    failures = 0
    stalled = 0
    while dataset is None or len(names_in(dataset)) < len(expected):
        previous_count = len(names_in(dataset)) if dataset else 0
        current = set(names_in(dataset)) if dataset else set()
        missing = [item for item in image_names if item not in current]
        batch = missing[:BATCH_SIZE]
        path = f"/datasets/{dataset['id']}/images" if dataset else "/datasets"
        try:
            result = upload_batch(client, path, image_dir, batch, creation_name=name if dataset is None else None, task=task)
            if dataset is None:
                dataset = result
                write_import_state(state_path, {**read_state(state_path), client.user_id: dataset["id"]})
            else:
                dataset = client.get_dataset(dataset["id"])
            failures = 0
        except requests.RequestException as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            if status is not None and status not in RETRY_STATUSES:
                raise RuntimeError(f"La API rechazó la subida de imágenes (HTTP {status})") from error
            failures += 1
            # Si la petición creó datos pero se perdió la respuesta, una nueva
            # consulta detecta esos archivos antes de intentar subirlos otra vez.
            print(f"Aviso: subida interrumpida; comprobando el estado remoto ({error})", flush=True)
            before = len(names_in(dataset)) if dataset else 0
            dataset = get_known_dataset(client, state_path, name, task, expected)
            if dataset and len(names_in(dataset)) > before:
                failures = 0
            if failures >= API_ATTEMPTS:
                raise RuntimeError("No se completó la subida; vuelve a ejecutar el comando para reanudar") from error
            pause(failures or 1)
        current_count = len(names_in(dataset)) if dataset else 0
        stalled = 0 if current_count > previous_count else stalled + 1
        if stalled >= API_ATTEMPTS:
            raise RuntimeError("La API no confirma el progreso de subida; vuelve a ejecutar el comando")
        if dataset:
            print(f"{task}: {current_count}/{len(expected)} imágenes importadas", flush=True)
    return dataset


def read_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("datasets", {})
    except (ValueError, OSError, AttributeError):
        return {}


def import_classification(client: ApiClient, result_dir: Path, rows: list[dict[str, str]]) -> str:
    name = f"TFG Ejemplo Imagenette {len(rows)} (seed 42)"
    dataset = import_images(client, result_dir, [row["filename"] for row in rows], name, "classification")
    expected_labels = {row["filename"]: row["label"] for row in rows}
    for index, image in enumerate(dataset["images"], 1):
        label = expected_labels[image["filename"]]
        if image.get("label") != label:
            client.request("PATCH", f"/datasets/{dataset['id']}/images/{image['id']}/label", json={"label": label})
        if index % 10 == 0 or index == len(rows):
            print(f"Clasificación: {index}/{len(rows)} etiquetas comprobadas", flush=True)
    dataset = client.get_dataset(dataset["id"])
    if len(dataset["images"]) != len(rows) or any(image.get("label") != expected_labels[image["filename"]] for image in dataset["images"]):
        raise RuntimeError("No se pudieron verificar todas las etiquetas de clasificación")
    return dataset["id"]


def import_detection(client: ApiClient, result_dir: Path, subset: dict) -> str:
    names = [image["file_name"] for image in subset["images"]]
    dataset = import_images(client, result_dir, names, f"TFG Ejemplo COCO {len(names)} (seed 2026)", "detection")
    dataset_id = dataset["id"]
    annotations_count = len(subset["annotations"])
    snapshot = result_dir / "instances_subset.json"
    coverage = client.request("GET", f"/datasets/{dataset_id}/annotations/coverage").json()
    # Verificación de cantidades permite reanudar si la importación se completó
    # pero se perdió la última respuesta HTTP.
    complete = (
        coverage.get("has_annotation_snapshot") is True
        and coverage.get("reviewed_images") == len(names)
        and coverage.get("total_objects") == annotations_count
        and coverage.get("total_categories") == len(subset["categories"])
    )
    if not complete:
        for attempt in range(1, API_ATTEMPTS + 1):
            try:
                with snapshot.open("rb") as handle:
                    response = client.session.post(
                        client.base + f"/datasets/{dataset_id}/annotations/coco",
                        files={"file": (snapshot.name, handle, "application/json")},
                        timeout=(15, 180),
                    )
                    response.raise_for_status()
                break
            except requests.RequestException as error:
                status = getattr(getattr(error, "response", None), "status_code", None)
                if status is not None and status not in RETRY_STATUSES:
                    raise RuntimeError(f"Importación COCO rechazada por la API (HTTP {status})") from error
                coverage = client.request("GET", f"/datasets/{dataset_id}/annotations/coverage").json()
                if (coverage.get("has_annotation_snapshot") and coverage.get("reviewed_images") == len(names)
                    and coverage.get("total_objects") == annotations_count):
                    break
                if attempt == API_ATTEMPTS:
                    raise RuntimeError("No se pudieron importar las anotaciones COCO") from error
                pause(attempt)
    coverage = client.request("GET", f"/datasets/{dataset_id}/annotations/coverage").json()
    if (not coverage.get("has_annotation_snapshot") or coverage.get("reviewed_images") != len(names)
        or coverage.get("total_objects") != annotations_count):
        raise RuntimeError("La cobertura COCO tras importar no coincide con el dataset")
    print(f"Detección: {annotations_count} objetos COCO verificados")
    return dataset_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Descarga y prepara datasets supervisados de ejemplo (10 a 100 imágenes)")
    parser.add_argument("--count", type=int, default=100, help="Imágenes por dataset (múltiplos de 10, máximo 100)")
    parser.add_argument("--task", choices=["both", "classification", "detection"], default="both")
    parser.add_argument("--prepare-only", action="store_true", help="No importar en la aplicación")
    parser.add_argument("--email", help="Correo de usuario registrado; contraseña solicitada sin mostrarla")
    parser.add_argument("--data-dir", type=Path, default=Path("/datasets"))
    parser.add_argument("--api-base", default="http://backend:8000")
    args = parser.parse_args(argv)
    validate_count(args.count)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    tasks = ["classification", "detection"] if args.task == "both" else [args.task]
    client = None
    if not args.prepare_only:
        print("Necesitas una cuenta creada previamente en http://localhost:5173")
        client = ApiClient(args.api_base, email=args.email)
    for task in tasks:
        if task == "classification":
            directory, rows = prepare_imagenette(args.data_dir, args.count)
            if client:
                dataset_id = import_classification(client, directory, rows)
                print(f"Clasificación disponible en la aplicación: dataset {dataset_id}")
        else:
            directory, subset = prepare_coco(args.data_dir, args.count)
            if client:
                dataset_id = import_detection(client, directory, subset)
                print(f"Detección disponible en la aplicación: dataset {dataset_id}")
    print("\nPreparación completada.")
    if args.prepare_only:
        print(f"Archivos y etiquetas/anotaciones: {args.data_dir}")
        print("Para importarlos automáticamente, repite el comando sin --prepare-only.")
    else:
        print("Los datasets ya tienen sus etiquetas o anotaciones y están listos para evaluar.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, requests.RequestException) as error:
        print(f"Error: {error}", file=sys.stderr)
        print("Puedes repetir el comando: se reutilizan los archivos y datasets ya preparados.", file=sys.stderr)
        raise SystemExit(1) from error
