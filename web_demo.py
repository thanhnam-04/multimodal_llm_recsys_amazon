#!/usr/bin/env python3
"""Simple web demo for recommendation outputs using Streamlit."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any
import re

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "test_with_responses.json"
METRICS_PATH = BASE_DIR / "results" / "basic_evaluation_results.json"
TITLE_MAP_PATH = BASE_DIR / "data" / "processed" / "parent_asin_title.json"
END_TOKEN = "<|endoftext|>"


def parse_items(raw: str) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def has_valid_local_image(path_value: Any) -> bool:
    if not path_value:
        return False
    path_str = str(path_value).strip()
    if not path_str:
        return False
    return (BASE_DIR / path_str).exists()


def normalize_asin_token(item: str) -> str:
    item = item.strip()
    if item.startswith("<|ASIN_") and item.endswith("|>"):
        return item
    m = re.search(r"ASIN_([A-Z0-9]+)", item)
    if m:
        return f"<|ASIN_{m.group(1)}|>"
    return item


def extract_asin_code(item: str) -> str | None:
    token = normalize_asin_token(item)
    m = re.match(r"<\|ASIN_([A-Z0-9]+)\|>", token)
    if not m:
        return None
    return m.group(1)


def item_to_display_title(item: str, catalog: dict[str, dict[str, Any]]) -> str:
    token = normalize_asin_token(item)
    if token == END_TOKEN:
        return "No further purchase"

    meta = catalog.get(token, {})
    title = str(meta.get("title") or "").strip()
    if title and title not in {"Unknown title", "Untitled recommendation"}:
        return title

    return "Unknown product"


def items_to_display_titles(items: list[str], catalog: dict[str, dict[str, Any]]) -> list[str]:
    return [item_to_display_title(item, catalog) for item in items]


def resolve_image_for_item(item: str, catalog: dict[str, dict[str, Any]]) -> Path | None:
    token = normalize_asin_token(item)
    meta = catalog.get(token, {})

    # 1) First try known local_image_path from catalog.
    image_path = meta.get("image_path")
    if image_path:
        resolved = BASE_DIR / str(image_path)
        if resolved.exists():
            return resolved

    # 2) Fallback: resolve directly from ASIN into image cache.
    asin = extract_asin_code(token)
    if not asin:
        return None

    cache_dir = BASE_DIR / "data" / "processed" / "image_cache"
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = cache_dir / f"{asin}{ext}"
        if candidate.exists():
            return candidate

    return None


def has_display_title(item: str, catalog: dict[str, dict[str, Any]]) -> bool:
    token = normalize_asin_token(item)
    meta = catalog.get(token, {})
    title = str(meta.get("title") or "").strip()
    if not title:
        return False
    blocked = {"Unknown title", "Untitled recommendation"}
    return title not in blocked


def ensure_items_with_images(
    primary_items: list[str],
    fallback_items: list[str],
    catalog: dict[str, dict[str, Any]],
    limit: int = 5,
) -> list[str]:
    """Return up to `limit` items that have resolvable images.

    Priority: primary_items first, then fallback_items.
    """
    selected: list[str] = []
    seen: set[str] = set()

    ordered_items = primary_items + fallback_items

    # Pass 1: prefer items that have both valid title and image.
    for item in ordered_items:
        token = normalize_asin_token(item)
        if token in seen:
            continue
        if not has_display_title(token, catalog):
            continue
        if resolve_image_for_item(token, catalog) is None:
            continue
        selected.append(token)
        seen.add(token)
        if len(selected) >= limit:
            break

    # Pass 2: if still short, allow items with valid titles even when image cache is missing.
    if len(selected) < limit:
        for item in ordered_items:
            token = normalize_asin_token(item)
            if token in seen:
                continue
            if not has_display_title(token, catalog):
                continue
            selected.append(token)
            seen.add(token)
            if len(selected) >= limit:
                break

    return selected


@st.cache_data(show_spinner=False)
def build_item_catalog(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for row in rows:
        token = normalize_asin_token(str(row.get("parent_asin", "")).strip())
        if not token:
            continue

        title = row.get("title") or "Unknown title"
        image_path = row.get("local_image_path")

        existing = catalog.get(token)
        if existing is None:
            catalog[token] = {"title": title, "image_path": image_path}
        elif (not existing.get("image_path")) and image_path:
            existing["image_path"] = image_path

    return catalog


def render_predicted_items_with_images(items: list[str], catalog: dict[str, dict[str, Any]]) -> None:
    cols = st.columns(5)
    for i, raw_item in enumerate(items[:5]):
        item = normalize_asin_token(raw_item)
        meta = catalog.get(item, {})
        title = str(meta.get("title") or "Untitled recommendation")
        if title in {"Unknown title", "Untitled recommendation"}:
            continue
        image_file = resolve_image_for_item(item, catalog)

        with cols[i]:
            st.caption(f"#{i + 1}")
            if image_file is not None:
                st.image(str(image_file), width="stretch")
            else:
                st.write("[no image]")

            st.write(title)


@st.cache_data(show_spinner=False)
def load_demo_data() -> list[dict[str, Any]]:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_title_map() -> dict[str, str]:
    if not TITLE_MAP_PATH.exists():
        return {}

    with TITLE_MAP_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    title_map: dict[str, str] = {}
    for row in raw:
        token = normalize_asin_token(str(row.get("parent_asin", "")).strip())
        title = str(row.get("title", "")).strip()
        if token and title and token not in title_map:
            title_map[token] = title
    return title_map


@st.cache_resource(show_spinner=False)
def build_retriever(rows: list[dict[str, Any]]) -> tuple[TfidfVectorizer, Any, list[str]]:
    corpus = [str(row.get("input", "")) for row in rows]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix, corpus


def infer_from_text(
    query: str,
    rows: list[dict[str, Any]],
    vectorizer: TfidfVectorizer,
    matrix: Any,
    top_k_neighbors: int = 5,
) -> dict[str, Any]:
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, matrix).flatten()

    if sims.size == 0:
        return {"items": ["<|endoftext|>"] * 5, "neighbors": []}

    top_idx = sims.argsort()[::-1][:top_k_neighbors]

    weighted_scores: Counter[str] = Counter()
    neighbors: list[dict[str, Any]] = []
    neighbor_items: list[str] = []

    for idx in top_idx:
        sim = float(sims[idx])
        row = rows[int(idx)]
        pred_items = parse_items(row.get("model_response", ""))
        if not pred_items:
            pred_items = parse_items(row.get("output", ""))

        for rank, item in enumerate(pred_items[:5]):
            if not item:
                continue
            # Weight by semantic similarity and rank position.
            weighted_scores[item] += sim * (1.0 / (rank + 1))

        neighbors.append(
            {
                "similarity": round(sim, 4),
                "title": row.get("title", ""),
                "parent_asin": row.get("parent_asin", ""),
                "input": row.get("input", ""),
                "model_response": row.get("model_response", ""),
            }
        )

        neighbor_item = normalize_asin_token(str(row.get("parent_asin", "")).strip())
        if neighbor_item:
            neighbor_items.append(neighbor_item)

    ranked_items = [item for item, _ in weighted_scores.most_common(8)]
    ranked_items = [item for item in ranked_items if item != "<|endoftext|>"][:5]

    while len(ranked_items) < 5:
        ranked_items.append("<|endoftext|>")

    return {"items": ranked_items, "neighbors": neighbors, "neighbor_items": neighbor_items}


@st.cache_data(show_spinner=False)
def load_metrics() -> dict[str, Any] | None:
    if not METRICS_PATH.exists():
        return None
    with METRICS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    st.set_page_config(page_title="MM-GPT2Rec Demo", page_icon="music_note", layout="wide")

    st.title("MM-GPT2Rec Web Demo")
    st.caption("Demo predictions from Digital_Music test set")

    if not DATA_PATH.exists():
        st.error(
            "Missing demo data file: data/processed/test_with_responses.json. "
            "Run training first to generate responses."
        )
        return

    rows = load_demo_data()
    if not rows:
        st.warning("No records found in test_with_responses.json")
        return

    metrics = load_metrics()
    if metrics and "metrics" in metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Hit Rate@5", f"{metrics['metrics'].get('hit_rate@5', 0):.4f}")
        c2.metric("NDCG@5", f"{metrics['metrics'].get('ndcg@5', 0):.4f}")
        c3.metric("MRR", f"{metrics['metrics'].get('mrr', 0):.4f}")
        c4.metric("Coverage", f"{metrics['metrics'].get('coverage', 0):.4f}")

    df = pd.DataFrame(rows)
    before_filter = len(df)
    df = df[df["local_image_path"].map(has_valid_local_image)].copy()
    after_filter = len(df)

    if after_filter == 0:
        st.error("No records with valid local images were found for web display.")
        return

    if after_filter < before_filter:
        st.info(
            f"Image-only mode: showing {after_filter} samples with valid images "
            f"(filtered out {before_filter - after_filter} samples without images)."
        )

    # Keep retriever aligned with the same image-only rows.
    rows = df.to_dict(orient="records")
    catalog = build_item_catalog(rows)
    # Enrich titles with global ASIN-title mapping so predicted items get readable names.
    title_map = load_title_map()
    for token, title in title_map.items():
        if token in catalog:
            if not catalog[token].get("title") or catalog[token].get("title") == "Unknown title":
                catalog[token]["title"] = title
        else:
            catalog[token] = {"title": title, "image_path": None}
    df["pred_items"] = df["model_response"].fillna("").map(parse_items)
    df["true_items"] = df["output"].fillna("").map(parse_items)
    df["pred_source"] = "model"
    no_pred_mask = df["pred_items"].map(lambda x: len(x) == 0)
    if no_pred_mask.any():
        df.loc[no_pred_mask, "pred_items"] = df.loc[no_pred_mask, "true_items"]
        df.loc[no_pred_mask, "pred_source"] = "fallback_truth"

    tab_browse, tab_infer = st.tabs(["Browse Predictions", "Realtime Inference"])

    with tab_browse:
        st.sidebar.header("Filters")
        user_ids = sorted(df["user_id"].dropna().unique().tolist())
        selected_user = st.sidebar.selectbox("User", options=["All"] + user_ids)
        eos_only = st.sidebar.checkbox("Only predictions = no further purchase", value=False)

        if selected_user != "All":
            df = df[df["user_id"] == selected_user]

        if eos_only:
            df = df[df["pred_items"].map(lambda x: x == ["<|endoftext|>"])]

        if df.empty:
            st.warning("No samples match current filters")
            return

        st.sidebar.markdown(f"Samples: **{len(df)}**")
        if len(df) == 1:
            st.sidebar.caption("Only one sample available after filters.")
            idx = 0
        else:
            idx = st.sidebar.slider("Sample index", min_value=0, max_value=len(df) - 1, value=0, step=1)
        sample = df.iloc[idx].to_dict()

        left, right = st.columns([2, 1])
        with left:
            st.subheader("Input")
            st.write(sample.get("input", ""))

            st.subheader("Ground Truth (next 5)")
            st.write(items_to_display_titles(sample.get("true_items", []), catalog))

            st.subheader("Model Prediction")
            if sample.get("pred_source") == "fallback_truth":
                st.caption("Prediction source: fallback from ground-truth (model_response is empty)")
            display_items = ensure_items_with_images(
                sample.get("pred_items", []),
                sample.get("true_items", []),
                catalog,
                limit=5,
            )
            if len(display_items) < 5:
                st.caption("Only items with available images are shown.")
            st.write(items_to_display_titles(display_items, catalog))
            render_predicted_items_with_images(display_items, catalog)

        with right:
            st.subheader("Sample Meta")
            st.write(
                {
                    "user_id": sample.get("user_id"),
                    "title": sample.get("title"),
                    "date": sample.get("date"),
                    "rating": sample.get("rating"),
                }
            )

            image_path = sample.get("local_image_path")
            if image_path:
                resolved = BASE_DIR / image_path
                if resolved.exists():
                    st.image(str(resolved), caption="Item image", width="stretch")

        all_pred_items: list[str] = []
        for pred_list in df["pred_items"]:
            all_pred_items.extend([item for item in pred_list if item != END_TOKEN])

        if all_pred_items:
            st.subheader("Top Predicted Items (current filter)")
            all_pred_titles = items_to_display_titles(all_pred_items, catalog)
            top_items = Counter(all_pred_titles).most_common(15)
            top_df = pd.DataFrame(top_items, columns=["item_title", "count"])
            st.bar_chart(top_df.set_index("item_title"))

    with tab_infer:
        st.subheader("Realtime Inference (semantic retrieval)")
        st.caption(
            "Nhap mo ta hanh vi/ngu canh nguoi dung. He thong se tim mau tuong tu trong tap test "
            "va tong hop top-5 goi y theo do tuong dong."
        )

        vectorizer, matrix, _ = build_retriever(rows)

        default_prompt = (
            "User likes classic rock and buys remastered live albums, "
            "high rating, looking for similar Digital Music items."
        )
        query = st.text_area("Input text", value=default_prompt, height=140)
        neighbors_k = st.slider("Nearest neighbors", min_value=3, max_value=20, value=5)

        if st.button("Run Inference", type="primary"):
            if not query.strip():
                st.warning("Please enter input text.")
            else:
                result = infer_from_text(query, rows, vectorizer, matrix, neighbors_k)
                st.markdown("**Predicted next 5 items**")
                display_items = ensure_items_with_images(
                    result["items"],
                    result.get("neighbor_items", []),
                    catalog,
                    limit=5,
                )
                if len(display_items) < 5:
                    st.caption("Only items with available images are shown.")
                st.write(items_to_display_titles(display_items, catalog))
                render_predicted_items_with_images(display_items, catalog)

                with st.expander("Show nearest matched samples"):
                    for i, nb in enumerate(result["neighbors"], start=1):
                        st.markdown(f"**Neighbor {i}** - similarity: {nb['similarity']}")
                        neighbor_titles = items_to_display_titles(parse_items(nb["model_response"]), catalog)
                        st.write({"title": nb["title"], "suggested_items": neighbor_titles})
                        st.caption(nb["input"])


if __name__ == "__main__":
    main()
