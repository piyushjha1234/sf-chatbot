"""JSON / YAML / OpenAPI / XML / source code -> Blocks."""

from __future__ import annotations

import json
from typing import Any

import yaml
from lxml import etree

from sfrag.core.models import Block, NormalizedDocument
from sfrag.pipeline.extract.base import ExtractionError

_HTTP_METHODS = ("get", "put", "post", "delete", "patch", "head", "options")


def _code(obj: Any, lang: str = "json") -> str:
    body = json.dumps(obj, indent=2, ensure_ascii=False, default=str) if lang == "json" else str(obj)
    return f"```{lang}\n{body}\n```"


def _openapi_blocks(spec: dict[str, Any]) -> tuple[str, list[Block]]:
    info = spec.get("info", {}) or {}
    title = info.get("title") or "API"
    version = info.get("version")
    blocks = [Block(type="heading", text=f"# {title}", level=1)]
    if info.get("description") or version:
        blocks.append(Block(type="paragraph", text=f"{info.get('description', '')} (version {version})".strip()))
    servers = [s.get("url") for s in spec.get("servers", []) if isinstance(s, dict)]
    if servers:
        blocks.append(Block(type="paragraph", text="Servers: " + ", ".join(f"`{s}`" for s in servers)))
    sec = spec.get("components", {}).get("securitySchemes") or spec.get("securityDefinitions")
    if sec:
        blocks.append(Block(type="heading", text="## Authentication", level=2))
        blocks.append(Block(type="code", text=_code(sec), language="json"))
    for path, item in (spec.get("paths") or {}).items():
        if not isinstance(item, dict):
            continue
        for method in _HTTP_METHODS:
            op = item.get(method)
            if not isinstance(op, dict):
                continue
            blocks.append(Block(type="heading", text=f"## {method.upper()} {path}", level=2))
            desc = " ".join(x for x in [op.get("summary"), op.get("description")] if x)
            if desc:
                blocks.append(Block(type="paragraph", text=desc.strip()))
            params = [*item.get("parameters", []), *op.get("parameters", [])]
            if params:
                header = "| Name | In | Required | Type | Description |\n|---|---|---|---|---|"
                rows = []
                for p in params:
                    if not isinstance(p, dict):
                        continue
                    schema = p.get("schema", {}) or {}
                    rows.append(
                        f"| `{p.get('name', p.get('$ref', ''))}` | {p.get('in', '')} | {p.get('required', False)} | "
                        f"{schema.get('type', p.get('type', ''))} | {(p.get('description') or '').replace('|', '/')} |"
                    )
                blocks.append(
                    Block(type="table", text="\n".join([header, *rows]), table_header=header, table_rows=rows)
                )
            if op.get("requestBody"):
                blocks.append(Block(type="paragraph", text="**Request body**"))
                blocks.append(Block(type="code", text=_code(op["requestBody"]), language="json"))
            if op.get("responses"):
                blocks.append(Block(type="paragraph", text="**Responses**"))
                blocks.append(Block(type="code", text=_code(op["responses"]), language="json"))
    schemas = (spec.get("components") or {}).get("schemas") or spec.get("definitions") or {}
    if schemas:
        blocks.append(Block(type="heading", text="## Schemas", level=2))
        for name, schema in schemas.items():
            blocks.append(Block(type="heading", text=f"### {name}", level=3))
            blocks.append(Block(type="code", text=_code(schema), language="json"))
    return title, blocks


def _generic_blocks(obj: Any, title: str) -> list[Block]:
    blocks = [Block(type="heading", text=f"# {title}", level=1)]
    if isinstance(obj, dict):
        for k, v in obj.items():
            blocks.append(Block(type="heading", text=f"## {k}", level=2))
            blocks.append(Block(type="code", text=_code(v), language="json"))
    else:
        blocks.append(Block(type="code", text=_code(obj), language="json"))
    return blocks


class JsonYamlParser:
    def parse(self, data: bytes, origin: str, mime: str) -> NormalizedDocument:
        text = data.decode("utf-8", errors="replace")
        try:
            obj = json.loads(text) if "json" in mime else yaml.safe_load(text)
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ExtractionError(f"malformed {mime}: {e}") from e
        name = origin.rstrip("/").rsplit("/", 1)[-1]
        if isinstance(obj, dict) and ("openapi" in obj or "swagger" in obj):
            title, blocks = _openapi_blocks(obj)
            return NormalizedDocument(title=title, blocks=blocks, origin=origin, mime=mime, meta={"openapi": True})
        return NormalizedDocument(title=name, blocks=_generic_blocks(obj, name), origin=origin, mime=mime)


class XmlParser:
    def parse(self, data: bytes, origin: str, mime: str) -> NormalizedDocument:
        try:
            parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)
            root = etree.fromstring(data, parser=parser)
        except etree.XMLSyntaxError as e:
            raise ExtractionError(f"malformed XML: {e}") from e
        name = origin.rstrip("/").rsplit("/", 1)[-1]
        blocks = [Block(type="heading", text=f"# {name}", level=1)]
        children = list(root)
        if not children:
            blocks.append(Block(type="code", text=f"```xml\n{etree.tostring(root, encoding=str)}\n```", language="xml"))
        for child in children:
            if not isinstance(child.tag, str):
                continue
            tag = etree.QName(child).localname
            blocks.append(Block(type="heading", text=f"## {tag}", level=2))
            blocks.append(
                Block(
                    type="code",
                    text=f"```xml\n{etree.tostring(child, encoding=str, pretty_print=True).strip()}\n```",
                    language="xml",
                )
            )
        return NormalizedDocument(title=name, blocks=blocks, origin=origin, mime=mime)


class CodeParser:
    def parse(self, data: bytes, origin: str, mime: str) -> NormalizedDocument:
        lang = mime.split(";", 1)[1] if ";" in mime else None
        text = data.decode("utf-8", errors="replace").rstrip("\n")
        name = origin.rstrip("/").rsplit("/", 1)[-1]
        blocks = [
            Block(type="heading", text=f"# {name}", level=1),
            Block(type="code", text=f"```{lang or ''}\n{text}\n```", language=lang),
        ]
        return NormalizedDocument(title=name, blocks=blocks, origin=origin, mime=mime.split(";")[0])
