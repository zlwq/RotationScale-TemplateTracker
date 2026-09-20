import cv2
import numpy as np
import argparse
import sys


# -------------------- basic preprocessing --------------------

def preprocess_gray(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def preprocess_edge(img):
    gray = preprocess_gray(img)
    edge = cv2.Canny(gray, 50, 150)
    return edge


# -------------------- rotate / scale template --------------------

def rotate_bound_with_matrix(img, angle, border_value=0):
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)

    M[0, 2] += nw / 2.0 - cx
    M[1, 2] += nh / 2.0 - cy

    rotated = cv2.warpAffine(img, M, (nw, nh), borderValue=border_value)
    return rotated, M


def rotate_bound(img, angle, border_value=0):
    rotated, _ = rotate_bound_with_matrix(img, angle, border_value)
    return rotated


# -------------------- target keypoints --------------------

def detect_target_keypoints(template, max_corners=120, quality=0.01, min_distance=5):
    gray = preprocess_gray(template)

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality,
        minDistance=min_distance,
        blockSize=3,
        useHarrisDetector=False
    )

    if corners is None:
        return []

    points = []
    for p in corners.reshape(-1, 2):
        points.append((float(p[0]), float(p[1])))

    return points


def parse_mark_kp(text, keypoints):
    if text is None or text.strip() == "":
        return []

    selected = []
    parts = text.replace(";", ",").split(",")

    for part in parts:
        part = part.strip()
        if part == "":
            continue
        idx = int(part)
        if idx < 0 or idx >= len(keypoints):
            raise RuntimeError(f"--mark-kp 里面的索引 {idx} 越界，当前关键点数量是 {len(keypoints)}")
        selected.append(idx)

    return selected


def parse_mark_xy(text):
    if text is None or text.strip() == "":
        return []

    points = []
    groups = text.split(";")

    for g in groups:
        g = g.strip()
        if g == "":
            continue
        xy = g.split(",")
        if len(xy) != 2:
            raise RuntimeError("--mark-xy 格式应为：x,y;x,y 例如：12,30;50,70")
        x = float(xy[0].strip())
        y = float(xy[1].strip())
        points.append((x, y))

    return points


def draw_keypoint_preview(template, keypoints, selected_indices=None):
    selected_indices = set(selected_indices or [])
    out = template.copy()

    for i, (x, y) in enumerate(keypoints):
        x = int(round(x))
        y = int(round(y))

        if i in selected_indices:
            color = (0, 0, 255)
            radius = 5
            thickness = 2
        else:
            color = (0, 255, 255)
            radius = 3
            thickness = 1

        cv2.circle(out, (x, y), radius, color, thickness)
        cv2.putText(
            out,
            str(i),
            (x + 4, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA
        )

    return out


def select_keypoints_interactive(template, keypoints, radius=25):
    if len(keypoints) == 0:
        return []

    selected = set()
    win = "select target keypoints: click to toggle, ENTER/SPACE to start, ESC/q to quit"

    def redraw():
        img = draw_keypoint_preview(template, keypoints, selected)
        cv2.putText(
            img,
            "click keypoints, ENTER/SPACE=start, ESC/q=quit",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )
        return img

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        pts = np.array(keypoints, dtype=np.float32)
        d = np.sqrt((pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2)
        idx = int(np.argmin(d))

        if d[idx] <= radius:
            if idx in selected:
                selected.remove(idx)
            else:
                selected.add(idx)
            cv2.imshow(win, redraw())
            print("selected keypoints:", sorted(selected))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    cv2.imshow(win, redraw())

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 10, 32):
            break
        if key in (27, ord("q"), ord("Q")):
            cv2.destroyWindow(win)
            raise RuntimeError("用户取消了关键点选择")

    cv2.destroyWindow(win)
    return sorted(selected)


# -------------------- template bank --------------------

def transform_points_for_template(selected_points, scale, rotate_matrix):
    offsets = []

    for x, y in selected_points:
        px = x * scale
        py = y * scale
        rx = rotate_matrix[0, 0] * px + rotate_matrix[0, 1] * py + rotate_matrix[0, 2]
        ry = rotate_matrix[1, 0] * px + rotate_matrix[1, 1] * py + rotate_matrix[1, 2]
        offsets.append((float(rx), float(ry)))

    return offsets


def build_template_bank(
    template,
    min_scale,
    max_scale,
    scale_step,
    angle_step,
    selected_points=None,
    min_template_size=8
):
    bank = []
    selected_points = selected_points or []

    tpl_gray_base = preprocess_gray(template)
    tpl_edge_base = preprocess_edge(template)

    gray_border = int(np.median(tpl_gray_base))

    scales = np.arange(min_scale, max_scale + 1e-6, scale_step)
    angles = np.arange(0, 181, angle_step)

    for scale in scales:
        tw = int(template.shape[1] * scale)
        th = int(template.shape[0] * scale)

        if tw < min_template_size or th < min_template_size:
            continue

        resized_gray = cv2.resize(
            tpl_gray_base,
            (tw, th),
            interpolation=cv2.INTER_LINEAR
        )

        resized_edge = cv2.resize(
            tpl_edge_base,
            (tw, th),
            interpolation=cv2.INTER_LINEAR
        )

        for angle in angles:
            rot_gray, M_gray = rotate_bound_with_matrix(resized_gray, angle, border_value=gray_border)
            rot_edge, _ = rotate_bound_with_matrix(resized_edge, angle, border_value=0)

            h, w = rot_gray.shape[:2]

            if w < min_template_size or h < min_template_size:
                continue

            mark_offsets = transform_points_for_template(selected_points, scale, M_gray)

            bank.append({
                "gray": rot_gray,
                "edge": rot_edge,
                "w": w,
                "h": h,
                "scale": float(scale),
                "angle": float(angle),
                "mark_offsets": mark_offsets
            })

    print("template bank size:", len(bank))
    return bank


# -------------------- match / NMS --------------------

def nms(boxes, scores, iou_thresh):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(1, x2 - x1 + 1) * np.maximum(1, y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        remain = np.where(iou <= iou_thresh)[0]
        order = order[remain + 1]

    return keep


def get_local_maxima(res, score_thresh, max_hits):
    mask = res >= score_thresh

    if not np.any(mask):
        return []

    kernel = np.ones((3, 3), np.uint8)
    local_max = res == cv2.dilate(res, kernel)
    mask = mask & local_max

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return []

    vals = res[ys, xs]

    if len(vals) > max_hits:
        idx = np.argpartition(vals, -max_hits)[-max_hits:]
        xs = xs[idx]
        ys = ys[idx]
        vals = vals[idx]

    order = np.argsort(vals)[::-1]

    hits = []
    for i in order:
        hits.append((int(xs[i]), int(ys[i]), float(vals[i])))

    return hits


def match_frame(
    frame,
    template_bank,
    score_thresh,
    nms_iou,
    top_k,
    max_hits_per_template,
    gray_weight,
    edge_weight
):
    frame_gray = preprocess_gray(frame)
    frame_edge = preprocess_edge(frame)

    H, W = frame_gray.shape[:2]

    boxes = []
    scores = []
    infos = []

    for item in template_bank:
        tpl_gray = item["gray"]
        tpl_edge = item["edge"]
        tw = item["w"]
        th = item["h"]

        if tw >= W or th >= H:
            continue

        res_gray = cv2.matchTemplate(
            frame_gray,
            tpl_gray,
            cv2.TM_CCOEFF_NORMED
        )

        res_edge = cv2.matchTemplate(
            frame_edge,
            tpl_edge,
            cv2.TM_CCOEFF_NORMED
        )

        res = gray_weight * res_gray + edge_weight * res_edge

        hits = get_local_maxima(
            res,
            score_thresh=score_thresh,
            max_hits=max_hits_per_template
        )

        for x, y, score in hits:
            boxes.append([x, y, x + tw, y + th])
            scores.append(score)
            infos.append({
                "scale": item["scale"],
                "angle": item["angle"],
                "mark_offsets": item["mark_offsets"]
            })

    if len(boxes) == 0:
        return []

    keep = nms(boxes, scores, iou_thresh=nms_iou)

    detections = []

    for idx in keep:
        detections.append({
            "box": boxes[idx],
            "score": scores[idx],
            "scale": infos[idx]["scale"],
            "angle": infos[idx]["angle"],
            "mark_offsets": infos[idx]["mark_offsets"]
        })

    detections.sort(key=lambda x: x["score"], reverse=True)
    return detections[:top_k]




# -------------------- jump-fix smoothing --------------------

def box_info(box):
    x1, y1, x2, y2 = box
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    cx = (float(x1) + float(x2)) / 2.0
    cy = (float(y1) + float(y2)) / 2.0
    return cx, cy, w, h


def pair_prev_curr(prev_dets, curr_dets):
    pairs = []
    used_prev = set()
    used_curr = set()
    candidates = []

    for pi, p in enumerate(prev_dets):
        pcx, pcy, pw, ph = box_info(p["box"])
        pdiag = (pw * pw + ph * ph) ** 0.5

        for ci, c in enumerate(curr_dets):
            ccx, ccy, cw, ch = box_info(c["box"])
            dist = ((pcx - ccx) ** 2 + (pcy - ccy) ** 2) ** 0.5
            wr = cw / pw
            hr = ch / ph

            if dist > pdiag * 1.0:
                continue
            if wr < 0.45 or wr > 2.2 or hr < 0.45 or hr > 2.2:
                continue

            size_cost = abs(cw - pw) / pw + abs(ch - ph) / ph
            cost = dist / pdiag + 0.3 * size_cost
            candidates.append((cost, pi, ci))

    candidates.sort(key=lambda x: x[0])

    for _, pi, ci in candidates:
        if pi in used_prev or ci in used_curr:
            continue
        used_prev.add(pi)
        used_curr.add(ci)
        pairs.append((pi, ci))

    return pairs


def jump_diff_and_thr(prev_det, curr_det):
    px1, py1, px2, py2 = prev_det["box"]
    cx1, cy1, cx2, cy2 = curr_det["box"]
    _, _, pw, ph = box_info(prev_det["box"])

    thr = max(8.0, 0.05 * max(pw, ph))
    diffs = [
        abs(float(cx1) - float(px1)),
        abs(float(cy1) - float(py1)),
        abs(float(cx2) - float(px2)),
        abs(float(cy2) - float(py2))
    ]
    diff = max(diffs)
    return diff, thr, diffs


def is_box_jump(prev_det, curr_det):
    diff, thr, _ = jump_diff_and_thr(prev_det, curr_det)
    return diff > thr


def union_search_rect(prev_det, curr_det, frame_w, frame_h, extra_pad_left=0.0, extra_pad_top=0.0, extra_pad_right=0.0, extra_pad_bottom=0.0):
    ax1, ay1, ax2, ay2 = [float(v) for v in prev_det["box"]]
    bx1, by1, bx2, by2 = [float(v) for v in curr_det["box"]]

    ux1 = min(ax1, bx1)
    uy1 = min(ay1, by1)
    ux2 = max(ax2, bx2)
    uy2 = max(ay2, by2)

    _, _, pw, ph = box_info(prev_det["box"])
    pad = max(15, 0.15 * max(pw, ph))

    rx1 = max(0, int(round(ux1 - pad - extra_pad_left)))
    ry1 = max(0, int(round(uy1 - pad - extra_pad_top)))
    rx2 = min(frame_w, int(round(ux2 + pad + extra_pad_right)))
    ry2 = min(frame_h, int(round(uy2 + pad + extra_pad_bottom)))

    return rx1, ry1, rx2, ry2


def random_jitter_pads(rng, jitter_px):
    if jitter_px <= 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        float(rng.uniform(0, jitter_px)),
        float(rng.uniform(0, jitter_px)),
        float(rng.uniform(0, jitter_px)),
        float(rng.uniform(0, jitter_px)),
    )


def local_refine_detection(
    frame,
    prev_det,
    curr_det,
    template_bank,
    score_thresh,
    gray_weight,
    edge_weight,
    frame_id=None,
    curr_index=None
):
    frame_gray = preprocess_gray(frame)
    frame_edge = preprocess_edge(frame)

    H, W = frame_gray.shape[:2]
    _, _, pw, ph = box_info(prev_det["box"])
    base_size = max(pw, ph)

    rng = np.random.default_rng()

    round_id = 1
    hit_limit = 80
    max_hit_limit = 40960
    max_rounds = 8
    best_ep_l = 0.0
    best_ep_t = 0.0
    best_ep_r = 0.0
    best_ep_b = 0.0
    prefix = f"frame {frame_id}: " if frame_id is not None else ""
    curr_tag = f"curr#{curr_index}" if curr_index is not None else "curr"

    while round_id <= max_rounds:
        jitter_px = float(round_id) * 3.0

        ep_l, ep_t, ep_r, ep_b = random_jitter_pads(rng, jitter_px)

        min_expand = 1.0

        best_ep_l += min_expand + ep_l
        best_ep_t += min_expand + ep_t
        best_ep_r += min_expand + ep_r
        best_ep_b += min_expand + ep_b

        rx1, ry1, rx2, ry2 = union_search_rect(
            prev_det,
            curr_det,
            W,
            H,
            extra_pad_left=best_ep_l,
            extra_pad_top=best_ep_t,
            extra_pad_right=best_ep_r,
            extra_pad_bottom=best_ep_b
        )



        roi_gray = frame_gray[ry1:ry2, rx1:rx2]
        roi_edge = frame_edge[ry1:ry2, rx1:rx2]

        if roi_gray.size == 0:
            return None

        best = None
        best_score = -1.0
        total_hits = 0
        rejected_by_thr = 0

        for item in template_bank:
            tw = item["w"]
            th = item["h"]

            if tw >= roi_gray.shape[1] or th >= roi_gray.shape[0]:
                continue

            if tw < pw * 0.55 or tw > pw * 1.8:
                continue
            if th < ph * 0.55 or th > ph * 1.8:
                continue

            res_gray = cv2.matchTemplate(
                roi_gray,
                item["gray"],
                cv2.TM_CCOEFF_NORMED
            )
            res_edge = cv2.matchTemplate(
                roi_edge,
                item["edge"],
                cv2.TM_CCOEFF_NORMED
            )
            res = gray_weight * res_gray + edge_weight * res_edge

            hits = get_local_maxima(
                res,
                score_thresh=score_thresh,
                max_hits=hit_limit
            )

            for lx, ly, score in hits:
                total_hits += 1
                gx = rx1 + lx
                gy = ry1 + ly
                candidate = {
                    "box": [gx, gy, gx + tw, gy + th],
                    "score": score,
                    "scale": item["scale"],
                    "angle": item["angle"],
                    "mark_offsets": item["mark_offsets"],
                    "refine_round": round_id,
                    "hit_limit": hit_limit,
                    "jitter_px": jitter_px,
                    "search_rect": [rx1, ry1, rx2, ry2]
                }

                if is_box_jump(prev_det, candidate):
                    rejected_by_thr += 1
                    continue

                if score > best_score:
                    best_score = score
                    best = candidate

        if best is not None:
            return best

        print(
            f"{prefix}REFINE JITTER ROUND {round_id}: no valid b1' under thr for {curr_tag}, "
            f"hit_limit={hit_limit}, total_hits={total_hits}, rejected_by_thr={rejected_by_thr}, "
            f"jitter_px={jitter_px:.1f}, search_rect={[rx1, ry1, rx2, ry2]}"
        )

        if hit_limit < max_hit_limit:
            hit_limit = min(max_hit_limit, hit_limit * 2)

        round_id += 1

    print(
        f"{prefix}REFINE STOP for {curr_tag}: reached max_rounds={max_rounds}, "
        f"max_hit_limit={max_hit_limit}, still no candidate satisfies score_thresh and thr"
    )
    return None

def fix_jumping_detections(
    frame,
    prev_dets,
    curr_dets,
    template_bank,
    score_thresh,
    gray_weight,
    edge_weight,
    frame_id=None
):
    if not prev_dets or not curr_dets:
        return curr_dets

    fixed = list(curr_dets)
    pairs = pair_prev_curr(prev_dets, curr_dets)

    for pi, ci in pairs:
        prev_det = prev_dets[pi]
        curr_det = fixed[ci]

        diff, thr, diffs = jump_diff_and_thr(prev_det, curr_det)
        if diff <= thr:
            continue

        prefix = f"frame {frame_id}: " if frame_id is not None else ""
        print(
            f"{prefix}JUMP prev#{pi} -> curr#{ci} "
            f"diff={diff:.1f} thr={thr:.1f} "
            f"diffs=[x1:{diffs[0]:.1f}, y1:{diffs[1]:.1f}, x2:{diffs[2]:.1f}, y2:{diffs[3]:.1f}] "
            f"prev_box={prev_det['box']} curr_box={curr_det['box']}"
        )

        refined = local_refine_detection(
            frame=frame,
            prev_det=prev_det,
            curr_det=curr_det,
            template_bank=template_bank,
            score_thresh=score_thresh,
            gray_weight=gray_weight,
            edge_weight=edge_weight,
            frame_id=frame_id,
            curr_index=ci
        )

        if refined is not None:
            new_diff, new_thr, _ = jump_diff_and_thr(prev_det, refined)
            fixed[ci] = refined
            print(
                f"{prefix}JUMP FIXED curr#{ci} "
                f"new_diff={new_diff:.1f} thr={new_thr:.1f} "
                f"score={refined['score']:.3f} "
                f"round={refined.get('refine_round', '?')} "
                f"hit_limit={refined.get('hit_limit', '?')} "
                f"jitter_px={refined.get('jitter_px', 0):.1f} "
                f"new_box={refined['box']}"
            )
        else:
            print(f"{prefix}JUMP NOT FIXED curr#{ci}: no candidate satisfies score_thresh and thr")

    return fixed

# -------------------- draw --------------------

def draw_target_mark(out, x, y, label=None):
    x = int(round(x))
    y = int(round(y))

    r = 5
    cv2.circle(out, (x, y), r, (0, 255, 255), 2)
    cv2.line(out, (x - r - 3, y), (x + r + 3, y), (0, 255, 255), 2)
    cv2.line(out, (x, y - r - 3), (x, y + r + 3), (0, 255, 255), 2)

    if label is not None:
        cv2.putText(
            out,
            str(label),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )


def draw_detections(frame, detections, show_text=True, show_selected_points=True):
    out = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        score = det["score"]

        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # cross = 10
        # cv2.line(out, (cx - cross, cy), (cx + cross, cy), (0, 0, 255), 2)
        # cv2.line(out, (cx, cy - cross), (cx, cy + cross), (0, 0, 255), 2)
        cv2.circle(out, (cx, cy), 3, (255, 0, 0), -1)

        if show_selected_points:
            for i, (ox, oy) in enumerate(det.get("mark_offsets", [])):
                px = x1 + ox
                py = y1 + oy
                draw_target_mark(out, px, py, label=i)

        if show_text:
            text = f"{score:.2f} a:{det['angle']:.0f} s:{det['scale']:.2f}"
            cv2.putText(
                out,
                text,
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    return out


# -------------------- main --------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=False)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", default="result.mp4")

    parser.add_argument("--min-scale", type=float, default=0.3)
    parser.add_argument("--max-scale", type=float, default=2.5)
    parser.add_argument("--scale-step", type=float, default=0.08)

    parser.add_argument("--angle-step", type=float, default=15)

    parser.add_argument("--score-thresh", type=float, default=0.28)
    parser.add_argument("--nms-iou", type=float, default=0.35)
    parser.add_argument("--top-k", type=int, default=30)

    parser.add_argument("--max-hits-per-template", type=int, default=20)

    parser.add_argument("--gray-weight", type=float, default=0.35)
    parser.add_argument("--edge-weight", type=float, default=0.65)

    parser.add_argument("--no-text", action="store_true")
    parser.add_argument("--no-selected-points", action="store_true")
    parser.add_argument("--smooth", action="store_true")

    parser.add_argument("--kp-preview", default="")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--select-kp", action="store_true")
    parser.add_argument("--mark-kp", default="")
    parser.add_argument("--mark-xy", default="")

    parser.add_argument("--kp-max-corners", type=int, default=120)
    parser.add_argument("--kp-quality", type=float, default=0.01)
    parser.add_argument("--kp-min-distance", type=float, default=5)
    parser.add_argument("--kp-select-radius", type=float, default=25)

    args = parser.parse_args()

    template = cv2.imread(args.target)
    if template is None:
        raise RuntimeError("target.jpg 读取失败")

    keypoints = detect_target_keypoints(
        template,
        max_corners=args.kp_max_corners,
        quality=args.kp_quality,
        min_distance=args.kp_min_distance
    )

    print("target keypoints:", len(keypoints))
    for i, (x, y) in enumerate(keypoints):
        print(f"kp {i}: x={x:.1f}, y={y:.1f}")

    selected_indices = parse_mark_kp(args.mark_kp, keypoints)

    if args.select_kp:
        selected_indices = select_keypoints_interactive(
            template,
            keypoints,
            radius=args.kp_select_radius
        )

    selected_points = [keypoints[i] for i in selected_indices]
    selected_points.extend(parse_mark_xy(args.mark_xy))

    print("selected target points:", len(selected_points))
    for i, (x, y) in enumerate(selected_points):
        print(f"selected {i}: x={x:.1f}, y={y:.1f}")

    if args.kp_preview != "":
        preview = draw_keypoint_preview(template, keypoints, selected_indices)
        cv2.imwrite(args.kp_preview, preview)
        print("saved keypoint preview:", args.kp_preview)

    if args.preview_only:
        return

    if not args.video:
        raise RuntimeError("没有传入 --video。只生成关键点预览时可以加 --preview-only。")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("mp4 视频读取失败")

    template_bank = build_template_bank(
        template=template,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        scale_step=args.scale_step,
        angle_step=args.angle_step,
        selected_points=selected_points
    )

    if len(template_bank) == 0:
        raise RuntimeError("模板库为空，检查 scale 参数或者 target.jpg 尺寸")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

    frame_id = 0
    prev_detections = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = match_frame(
            frame=frame,
            template_bank=template_bank,
            score_thresh=args.score_thresh,
            nms_iou=args.nms_iou,
            top_k=args.top_k,
            max_hits_per_template=args.max_hits_per_template,
            gray_weight=args.gray_weight,
            edge_weight=args.edge_weight
        )

        if args.smooth and prev_detections is not None:
            detections = fix_jumping_detections(
                frame=frame,
                prev_dets=prev_detections,
                curr_dets=detections,
                template_bank=template_bank,
                score_thresh=args.score_thresh,
                gray_weight=args.gray_weight,
                edge_weight=args.edge_weight,
                frame_id=frame_id
            )

        marked = draw_detections(
            frame,
            detections,
            show_text=not args.no_text,
            show_selected_points=not args.no_selected_points
        )

        writer.write(marked)

        prev_detections = detections
        frame_id += 1
        print(f"frame {frame_id}: detected {len(detections)}")

    cap.release()
    writer.release()

    print("done:", args.out)


if __name__ == "__main__":
    main()
