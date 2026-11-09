"""Qdrant vector store: named dense + sparse vectors, server-side RRF, generation collections behind an alias.

Embedded (on-disk) mode locks its directory to one process. Run the API with `--with-worker`, or point
`index.qdrant_url` at a Qdrant server when the worker and API run as separate processes.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import re
import threading
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from sfrag.core.settings import Settings
from sfrag.index.embedder import Embedding

log = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "sparse"

PAYLOAD_INDEXES: dict[str, qm.PayloadSchemaType] = {
    "doc_id": qm.PayloadSchemaType.KEYWORD,
    "parent_id": qm.PayloadSchemaType.KEYWORD,
    "products": qm.PayloadSchemaType.KEYWORD,
    "categories": qm.PayloadSchemaType.KEYWORD,
    "content_type": qm.PayloadSchemaType.KEYWORD,
    "release": qm.PayloadSchemaType.KEYWORD,
    "lifecycle": qm.PayloadSchemaType.KEYWORD,
    "acl": qm.PayloadSchemaType.KEYWORD,
    "identifiers": qm.PayloadSchemaType.KEYWORD,
    "source_id": qm.PayloadSchemaType.KEYWORD,
}

_clients: dict[str, QdrantClient] = {}
_clients_lock = threading.Lock()


def _client_for(settings: Settings) -> QdrantClient:
    key = settings.index.qdrant_url or str(settings.qdrant_location)
    with _clients_lock:
        if key not in _clients:
            if settings.index.qdrant_url:
                _clients[key] = QdrantClient(url=settings.index.qdrant_url, timeout=60)
            else:
                settings.qdrant_location.mkdir(parents=True, exist_ok=True)
                _clients[key] = QdrantClient(path=str(settings.qdrant_location))
        return _clients[key]


def close_clients() -> None:
    with _clients_lock:
        for c in _clients.values():
            with contextlib.suppress(Exception):
                c.close()
        _clients.clear()


atexit.register(close_clients)


class _Locked:
    """Proxy that serialises all method calls on the wrapped object."""

    def __init__(self, inner: Any, lock: threading.RLock) -> None:
        self._inner = inner
        self._lock = lock

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def call(*a: Any, **kw: Any) -> Any:
            with self._lock:
                return attr(*a, **kw)

        return call


_local_locks: dict[str, threading.RLock] = {}
_ensure_lock = threading.Lock()


def _local_lock(settings: Settings) -> threading.RLock:
    key = str(settings.qdrant_location)
    with _clients_lock:
        return _local_locks.setdefault(key, threading.RLock())


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        raw_client = _client_for(settings)
        self.alias = settings.index.alias
        self.prefix = settings.index.collection_prefix
        self.is_server = bool(settings.index.qdrant_url)
        # Embedded (local) mode is not safe for concurrent access from threads: serialise every call.
        self.client: QdrantClient = raw_client if self.is_server else _Locked(raw_client, _local_lock(settings))  # type: ignore[assignment]

    # ---- generations / alias ---------------------------------------------------------
    def live_collection(self) -> str | None:
        for a in self.client.get_aliases().aliases:
            if a.alias_name == self.alias:
                return a.collection_name
        return None

    def next_generation_name(self) -> str:
        nums = [
            int(m.group(1))
            for c in self.client.get_collections().collections
            if (m := re.fullmatch(re.escape(self.prefix) + r"(\d+)", c.name))
        ]
        return f"{self.prefix}{max(nums, default=0) + 1}"

    def create_collection(self, name: str, dim: int, sparse_idf: bool) -> None:
        quant = None
        if self.is_server and self.settings.index.quantization:
            quant = qm.ScalarQuantization(
                scalar=qm.ScalarQuantizationConfig(type=qm.ScalarType.INT8, quantile=0.99, always_ram=True)
            )
        self.client.create_collection(
            collection_name=name,
            vectors_config={DENSE: qm.VectorParams(size=dim, distance=qm.Distance.COSINE)},
            sparse_vectors_config={SPARSE: qm.SparseVectorParams(modifier=qm.Modifier.IDF if sparse_idf else None)},
            quantization_config=quant,
        )
        if self.is_server:
            for field, schema in PAYLOAD_INDEXES.items():
                self.client.create_payload_index(name, field_name=field, field_schema=schema)

    def ensure_live(self, dim: int, sparse_idf: bool) -> str:
        with _ensure_lock:
            live = self.live_collection()
            if live:
                return live
            name = self.next_generation_name()
            self.create_collection(name, dim, sparse_idf)
            self.swap_alias(name)
            return name

    def swap_alias(self, new_collection: str) -> str | None:
        """Atomically point the live alias at `new_collection`. Returns the previous collection."""
        old = self.live_collection()
        ops: list[Any] = []
        if old:
            ops.append(qm.DeleteAliasOperation(delete_alias=qm.DeleteAlias(alias_name=self.alias)))
        ops.append(
            qm.CreateAliasOperation(create_alias=qm.CreateAlias(collection_name=new_collection, alias_name=self.alias))
        )
        self.client.update_collection_aliases(change_aliases_operations=ops)
        return old

    def drop_collection(self, name: str) -> None:
        if self.client.collection_exists(name):
            self.client.delete_collection(name)

    def count(self, collection: str | None = None) -> int:
        coll = collection or self.alias
        return int(self.client.count(coll, exact=True).count)

    # ---- writes ------------------------------------------------------------------------
    def upsert(self, collection: str, points: list[tuple[str, Embedding, dict[str, Any]]]) -> None:
        if not points:
            return
        structs = [
            qm.PointStruct(
                id=pid,
                vector={
                    DENSE: emb.dense,
                    SPARSE: qm.SparseVector(indices=emb.sparse.indices, values=emb.sparse.values),
                },
                payload=payload,
            )
            for pid, emb, payload in points
        ]
        for i in range(0, len(structs), 256):
            self.client.upsert(collection_name=collection, points=structs[i : i + 256], wait=True)

    def delete_points(self, collection: str, point_ids: list[str]) -> None:
        if point_ids:
            self.client.delete(collection, points_selector=qm.PointIdsList(points=point_ids), wait=True)  # type: ignore[arg-type]

    def delete_doc(self, collection: str, doc_id: str) -> None:
        self.client.delete(
            collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))])
            ),
            wait=True,
        )

    # ---- reads ---------------------------------------------------------------------------
    def hybrid_search(
        self,
        query: Embedding,
        identifiers: list[str],
        flt: qm.Filter | None,
        dense_k: int,
        sparse_k: int,
        identifier_k: int,
        limit: int,
        collection: str | None = None,
    ) -> list[qm.ScoredPoint]:
        coll = collection or self.alias
        prefetch = [
            qm.Prefetch(query=query.dense, using=DENSE, filter=flt, limit=dense_k),
            qm.Prefetch(
                query=qm.SparseVector(indices=query.sparse.indices, values=query.sparse.values),
                using=SPARSE,
                filter=flt,
                limit=sparse_k,
            ),
        ]
        if identifiers:
            id_filter = qm.Filter(
                must=[*(flt.must if flt and flt.must else [])],
                must_not=flt.must_not if flt else None,
                should=[qm.FieldCondition(key="identifiers", match=qm.MatchAny(any=identifiers))],
            )
            prefetch.append(qm.Prefetch(query=query.dense, using=DENSE, filter=id_filter, limit=identifier_k))
        res = self.client.query_points(
            collection_name=coll,
            prefetch=prefetch,
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return list(res.points)
