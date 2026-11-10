// A3: decode the raw YOLOv8 output, equal to Ultralytics 8.4.208 non_max_suppression
// (multi_label=False, agnostic=False) + scale_boxes + clip_boxes. Pure function.
import type { Box } from "./types";

export interface DecodeOptions {
  conf: number;   // keep anchors whose best class score is > conf
  iou: number;    // same-class boxes with IoU > iou are suppressed
  maxDet: number; // keep at most this many boxes
}

/** Where the original image sits inside the letterboxed input (see letterbox()). */
export interface DecodeMeta {
  left: number;
  top: number;
  gx: number;     // nw / width
  gy: number;     // nh / height
  width: number;  // original image size, boxes are clipped to it
  height: number;
}

export function iou(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const iw = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const ih = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = iw * ih;
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}

/**
 * SPEC §6.3. `output` is output0 flattened, channel-major [1, 4 + C, N]: value k of anchor i is
 * output[k * N + i]; k = 0..3 are cx, cy, w, h in input pixels, k = 4.. are class scores.
 * Returns boxes in ORIGINAL image pixels, sorted by conf descending.
 */
export function decode(output: ArrayLike<number>, C: number, opts: DecodeOptions, meta: DecodeMeta, labels?: string[]): Box[] {
  const N = output.length / (4 + C);
  if (!Number.isInteger(N) || N <= 0) throw new Error(`decode: output length ${output.length} is not (4 + ${C}) * N`);

  // 1. best class per anchor, keep conf > threshold
  const cand: Array<{ i: number; cls: number; conf: number; box: [number, number, number, number] }> = [];
  for (let i = 0; i < N; i++) {
    let best = -Infinity, cls = 0;
    for (let c = 0; c < C; c++) {
      const s = output[(4 + c) * N + i];
      if (s > best) { best = s; cls = c; }
    }
    if (!(best > opts.conf)) continue;
    // 2. cxcywh -> xyxy
    const cx = output[i], cy = output[N + i], w = output[2 * N + i], h = output[3 * N + i];
    cand.push({ i, cls, conf: best, box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2] });
  }
  // 3. NMS per class, greedy in conf order (= Ultralytics' class-offset trick + torchvision nms)
  cand.sort((a, b) => b.conf - a.conf || a.i - b.i);
  const keptByClass = new Map<number, Array<[number, number, number, number]>>();
  const kept: typeof cand = [];
  for (const c of cand) {
    if (kept.length >= opts.maxDet) break; // 4. max_det
    const same = keptByClass.get(c.cls) ?? [];
    if (same.some((k) => iou(k, c.box) > opts.iou)) continue;
    same.push(c.box);
    keptByClass.set(c.cls, same);
    kept.push(c);
  }
  // 5. undo letterbox, clip to the original image
  const clip = (v: number, hi: number) => (v < 0 ? 0 : v > hi ? hi : v);
  return kept.map((k) => {
    const [x1, y1, x2, y2] = k.box;
    return {
      cls: labels?.[k.cls] ?? String(k.cls),
      classId: k.cls,
      conf: k.conf,
      xyxy: [
        clip((x1 - meta.left) / meta.gx, meta.width),
        clip((y1 - meta.top) / meta.gy, meta.height),
        clip((x2 - meta.left) / meta.gx, meta.width),
        clip((y2 - meta.top) / meta.gy, meta.height),
      ],
    };
  });
}
