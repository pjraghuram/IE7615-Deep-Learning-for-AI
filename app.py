import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "yolov8s_milestone2_best.pt"

# Inference always runs at this floor so the app can show *everything* the model
# scored, including the boxes the confidence slider is currently throwing away.
SCAN_CONF = 0.01
OVERLAP_FOR_CLASS_SPREAD = 0.5

PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#B279A2", "#FF9DA6", "#9D755D", "#EECA3B", "#BAB0AC",
]

st.set_page_config(page_title="YOLOv8s Object Detection", layout="wide")
st.title("YOLOv8s Object Detection")


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


model = load_model()


@st.cache_data(show_spinner="Running inference...")
def run_inference(image_bytes: bytes, iou: float):
    """Single low-confidence pass. Cached on (image, iou) so moving the confidence
    slider only re-filters the tables instead of re-running YOLO."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    boxes = model.predict(image, conf=SCAN_CONF, iou=iou, verbose=False)[0].boxes
    if len(boxes) == 0:
        return (
            np.zeros((0, 4), dtype=float),
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=int),
        )
    return (
        boxes.xyxy.cpu().numpy().astype(float),
        boxes.conf.cpu().numpy().astype(float),
        boxes.cls.cpu().numpy().astype(int),
    )


def pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def draw_boxes(image, boxes, confs, classes, names):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    stroke = max(2, round(max(canvas.size) / 400))
    try:
        font = ImageFont.load_default(size=max(14, round(max(canvas.size) / 55)))
    except TypeError:  # Pillow < 10.1 takes no size argument
        font = ImageFont.load_default()
    text_h = getattr(font, "size", 11)

    for i, ((x1, y1, x2, y2), c, k) in enumerate(zip(boxes, confs, classes)):
        color = PALETTE[int(k) % len(PALETTE)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=stroke)
        label = f"#{i} {names[int(k)]} {c:.2f}"
        anchor = (x1, max(0.0, y1 - text_h - 4))
        tx1, ty1, tx2, ty2 = draw.textbbox(anchor, label, font=font)
        draw.rectangle([tx1 - 2, ty1 - 2, tx2 + 2, ty2 + 2], fill=color)
        draw.text(anchor, label, fill="white", font=font)
    return canvas


st.sidebar.header("Thresholds")
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
iou = st.sidebar.slider("IoU (NMS) threshold", 0.1, 0.9, 0.45, 0.05)
st.sidebar.caption(
    f"Inference runs at conf={SCAN_CONF} and the slider filters afterwards, so the "
    "tables also show the boxes you are discarding."
)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded is None:
    st.info("Upload an image to see the prediction and the per-box score tables.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
img_w, img_h = image.size
img_area = float(img_w * img_h)
names = model.names

boxes_all, confs_all, cls_all = run_inference(uploaded.getvalue(), iou)

if len(boxes_all) == 0:
    st.image(image, caption="Original", use_container_width=True)
    st.warning(f"The model scored nothing above {SCAN_CONF}. There is no threshold to tune here.")
    st.stop()

order = np.argsort(-confs_all)
boxes_all, confs_all, cls_all = boxes_all[order], confs_all[order], cls_all[order]

classes_present = sorted({names[int(k)] for k in cls_all})
selected = st.sidebar.multiselect("Classes", classes_present, default=classes_present)
class_mask = np.array([names[int(k)] in selected for k in cls_all])

keep_mask = (confs_all >= conf) & class_mask
boxes, confs, classes = boxes_all[keep_mask], confs_all[keep_mask], cls_all[keep_mask]

# Max overlap of each kept box against the other kept boxes: a value near the IoU
# slider marks a box that NMS almost suppressed (or almost kept).
iou_kept = pairwise_iou(boxes, boxes)
np.fill_diagonal(iou_kept, 0.0)
max_iou = iou_kept.max(axis=1) if len(boxes) > 1 else np.zeros(len(boxes))

kept_rows = pd.DataFrame(
    {
        "#": np.arange(len(boxes)),
        "class": [names[int(k)] for k in classes],
        "confidence": confs.round(4),
        "margin vs conf": (confs - conf).round(4),
        "max IoU w/ other box": max_iou.round(4),
        "x1": np.round(boxes[:, 0]).astype(int),
        "y1": np.round(boxes[:, 1]).astype(int),
        "x2": np.round(boxes[:, 2]).astype(int),
        "y2": np.round(boxes[:, 3]).astype(int),
        "w": np.round(boxes[:, 2] - boxes[:, 0]).astype(int),
        "h": np.round(boxes[:, 3] - boxes[:, 1]).astype(int),
        "area %": np.round(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / img_area * 100, 2
        ),
    }
)

conf_col = {
    "confidence": st.column_config.ProgressColumn(
        "confidence", min_value=0.0, max_value=1.0, format="%.3f"
    )
}

col_img, col_tbl = st.columns([1.15, 1], gap="medium")

with col_img:
    st.subheader("Prediction")
    st.image(
        draw_boxes(image, boxes, confs, classes, names),
        caption=f"conf >= {conf:.2f}, NMS IoU = {iou:.2f}",
        use_container_width=True,
    )
    with st.expander("Original image"):
        st.image(image, use_container_width=True)

with col_tbl:
    st.subheader(f"Detections kept ({len(boxes)})")
    m1, m2, m3 = st.columns(3)
    m1.metric("Kept", len(boxes))
    m2.metric("Filtered out", int(len(boxes_all) - len(boxes)))
    m3.metric("Top score", f"{confs_all.max():.3f}")
    m4, m5, m6 = st.columns(3)
    m4.metric("Mean conf (kept)", f"{confs.mean():.3f}" if len(confs) else "-")
    m5.metric("Weakest kept", f"{confs.min():.3f}" if len(confs) else "-")
    m6.metric("Max pair IoU", f"{max_iou.max():.3f}" if len(boxes) > 1 else "0.000")

    if len(boxes):
        st.dataframe(
            kept_rows,
            use_container_width=True,
            hide_index=True,
            height=min(430, 60 + 35 * len(kept_rows)),
            column_config=conf_col,
        )
        st.caption(
            "`margin vs conf` near 0 means that box appears or vanishes if you nudge the "
            "confidence slider. `max IoU w/ other box` near the IoU slider means the box "
            "is on the edge of being merged away by NMS."
        )
    else:
        st.info("Nothing is above the current confidence threshold - see the tables below.")

st.divider()

tab_all, tab_sweep, tab_iou, tab_spread = st.tabs(
    ["All scored boxes", "Threshold sweep", "IoU overlap matrix", "Class score spread"]
)

with tab_all:
    st.markdown(
        f"Every box the model produced at conf >= {SCAN_CONF}, after NMS at IoU = {iou:.2f}."
    )
    areas_all = (boxes_all[:, 2] - boxes_all[:, 0]) * (boxes_all[:, 3] - boxes_all[:, 1])
    all_rows = pd.DataFrame(
        {
            "class": [names[int(k)] for k in cls_all],
            "confidence": confs_all.round(4),
            "status": np.where(
                keep_mask, "kept", np.where(class_mask, "below conf", "class filtered")
            ),
            "margin vs conf": (confs_all - conf).round(4),
            "x1": np.round(boxes_all[:, 0]).astype(int),
            "y1": np.round(boxes_all[:, 1]).astype(int),
            "x2": np.round(boxes_all[:, 2]).astype(int),
            "y2": np.round(boxes_all[:, 3]).astype(int),
            "area %": np.round(areas_all / img_area * 100, 2),
        }
    )
    st.dataframe(all_rows, use_container_width=True, hide_index=True, column_config=conf_col)
    st.download_button(
        "Download scores as CSV",
        all_rows.to_csv(index=False).encode(),
        file_name=f"{Path(uploaded.name).stem}_scores.csv",
        mime="text/csv",
    )

    st.markdown("**Score distribution per class**")
    per_class = (
        all_rows.groupby("class")["confidence"]
        .agg(boxes="count", max="max", mean="mean", min="min")
        .round(4)
        .reset_index()
    )
    per_class["kept at current conf"] = [
        int(((all_rows["class"] == c) & (all_rows["status"] == "kept")).sum())
        for c in per_class["class"]
    ]
    st.dataframe(per_class, use_container_width=True, hide_index=True)

with tab_sweep:
    st.markdown(
        "How many boxes survive at each confidence threshold. A long flat stretch is a safe "
        "place to sit; a cliff means the threshold is doing real work right there."
    )
    steps = np.round(np.arange(0.05, 0.96, 0.05), 2)
    sweep = pd.DataFrame(
        {
            "conf threshold": steps,
            "detections": [int((confs_all >= t).sum()) for t in steps],
            "classes": [int(len(set(cls_all[confs_all >= t].tolist()))) for t in steps],
            "mean conf": [
                round(float(confs_all[confs_all >= t].mean()), 4)
                if (confs_all >= t).any()
                else 0.0
                for t in steps
            ],
        }
    )
    sweep["current"] = np.where(np.isclose(sweep["conf threshold"], conf), "<--", "")
    left, right = st.columns([1, 1.2])
    left.dataframe(sweep, use_container_width=True, hide_index=True, height=430)
    right.line_chart(sweep.set_index("conf threshold")["detections"])

    st.markdown("**Detections per class at each threshold** (strongest class first)")
    ranked_ids = sorted(
        set(cls_all.tolist()), key=lambda k: -float(confs_all[cls_all == k].max())
    )
    pivot = pd.DataFrame(
        {
            names[int(k)]: [int(((confs_all >= t) & (cls_all == k)).sum()) for t in steps]
            for k in ranked_ids
        },
        index=steps,
    )
    pivot.index.name = "conf threshold"
    st.dataframe(pivot, use_container_width=True)

with tab_iou:
    if len(boxes) < 2:
        st.info("Need at least two kept boxes to compare overlaps.")
    else:
        labels = [
            f"#{i} {names[int(k)]} {c:.2f}" for i, (k, c) in enumerate(zip(classes, confs))
        ]
        matrix = pd.DataFrame(iou_kept.round(3), index=labels, columns=labels)
        st.markdown(
            f"Pairwise IoU between the kept boxes. NMS only suppresses a same-class pair "
            f"above the IoU slider ({iou:.2f}), so pairs above it here are different classes."
        )
        try:
            st.dataframe(
                matrix.style.background_gradient(cmap="Reds", vmin=0, vmax=1).format("{:.3f}"),
                use_container_width=True,
            )
        except ImportError:
            st.dataframe(matrix, use_container_width=True)

        pairs = [
            {
                "box A": labels[i],
                "box B": labels[j],
                "IoU": round(float(iou_kept[i, j]), 4),
                "same class": bool(classes[i] == classes[j]),
                "IoU - slider": round(float(iou_kept[i, j] - iou), 4),
            }
            for i in range(len(boxes))
            for j in range(i + 1, len(boxes))
            if iou_kept[i, j] > 0
        ]
        if pairs:
            st.markdown("**Overlapping pairs**")
            st.dataframe(
                pd.DataFrame(pairs).sort_values("IoU", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No kept boxes overlap at all, so the IoU threshold is not biting here.")

with tab_spread:
    if len(boxes) == 0:
        st.info("No kept boxes to break down.")
    else:
        st.markdown(
            f"For each kept box, the best score of every class among the raw boxes that "
            f"overlap it (IoU >= {OVERLAP_FOR_CLASS_SPREAD}). Two classes with close scores "
            "on the same object means the model is unsure which label it is, and raising the "
            "confidence threshold will not fix that."
        )
        overlap = pairwise_iou(boxes, boxes_all)
        class_ids = sorted(set(cls_all.tolist()))
        spread = pd.DataFrame(
            0.0,
            index=[f"#{i} {names[int(k)]}" for i, k in enumerate(classes)],
            columns=[names[int(k)] for k in class_ids],
        )
        for i in range(len(boxes)):
            near = overlap[i] >= OVERLAP_FOR_CLASS_SPREAD
            for col, k in enumerate(class_ids):
                sel = near & (cls_all == k)
                if sel.any():
                    spread.iloc[i, col] = round(float(confs_all[sel].max()), 4)
        # A 73-class model scores a lot of nothing: keep only the classes that
        # actually landed on a kept box, strongest first.
        spread = spread.loc[:, spread.max(axis=0) > 0]
        spread = spread[spread.max(axis=0).sort_values(ascending=False).index]
        try:
            st.dataframe(
                spread.style.background_gradient(cmap="Blues", axis=1, vmin=0, vmax=1).format(
                    "{:.3f}"
                ),
                use_container_width=True,
            )
        except ImportError:
            st.dataframe(spread, use_container_width=True)
        st.caption(
            "These are per-box class scores, not a softmax - YOLO scores each class "
            "independently, so a row does not sum to 1."
        )
