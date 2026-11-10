// A2: letterbox preprocess, equal to Ultralytics 8.4.208 LetterBox(auto=False, center=True, scaleup=True)
// followed by BGR->RGB, /255, HWC->CHW. Pure function: no DOM, runs in node tests too.

/** Python's round(): ties go to the even integer (round(2.5) == 2, round(3.5) == 4). */
export function roundHalfEven(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

export interface Letterbox {
  tensor: Float32Array; // [1, 3, imgsz, imgsz] RGB / 255, channel-major
  r: number;            // scale ratio new / old (same for both axes before rounding)
  nw: number;           // resized width  = round(w * r)
  nh: number;           // resized height = round(h * r)
  left: number;         // padding on the left, in pixels of the imgsz canvas
  top: number;          // padding on the top
  gx: number;           // effective x gain nw / w (what Ultralytics scale_boxes divides by)
  gy: number;           // effective y gain nh / h
  imgsz: number;
}

/** Geometry only (no pixels), so it can be unit-tested against Python numbers. */
export function letterboxParams(w: number, h: number, imgsz = 640) {
  const r = Math.min(imgsz / h, imgsz / w);
  const nw = Math.max(roundHalfEven(w * r), 1);
  const nh = Math.max(roundHalfEven(h * r), 1);
  const dw = (imgsz - nw) / 2;
  const dh = (imgsz - nh) / 2;
  const left = roundHalfEven(dw - 0.1);
  const top = roundHalfEven(dh - 0.1);
  return { r, nw, nh, left, top, gx: nw / w, gy: nh / h };
}

const COEF_BITS = 11; // OpenCV INTER_RESIZE_COEF_BITS
const COEF_SCALE = 1 << COEF_BITS;

/** cv2.resize INTER_LINEAR tables for one axis: source index + fixed-point weights (sum 2048). */
function linearTable(src: number, dst: number): { idx: Int32Array; w0: Int32Array } {
  const scale = src / dst;
  const idx = new Int32Array(dst);
  const w0 = new Int32Array(dst);
  for (let d = 0; d < dst; d++) {
    let f = (d + 0.5) * scale - 0.5;
    let s = Math.floor(f);
    f -= s;
    if (s < 0) { f = 0; s = 0; }
    if (s >= src - 1) { f = 0; s = src - 1; }
    idx[d] = s;
    // saturate_cast<short>((1 - f) * 2048): OpenCV rounds to nearest
    w0[d] = Math.round((1 - f) * COEF_SCALE);
  }
  return { idx, w0 };
}

/** Bilinear resize of an RGBA buffer into RGB (3 channels), matching cv2 INTER_LINEAR for uint8. */
export function resizeBilinearRGB(rgba: ArrayLike<number>, w: number, h: number, nw: number, nh: number): Uint8Array {
  const out = new Uint8Array(nw * nh * 3);
  if (nw === w && nh === h) {
    for (let i = 0, j = 0; i < w * h; i++) {
      out[j++] = rgba[i * 4];
      out[j++] = rgba[i * 4 + 1];
      out[j++] = rgba[i * 4 + 2];
    }
    return out;
  }
  const xt = linearTable(w, nw);
  const yt = linearTable(h, nh);
  // horizontal pass for every source row actually used, kept as ints (value * 2048)
  const rowCache = new Map<number, Int32Array>();
  const hrow = (y: number): Int32Array => {
    let row = rowCache.get(y);
    if (row) return row;
    row = new Int32Array(nw * 3);
    const base = y * w * 4;
    for (let x = 0; x < nw; x++) {
      const s = xt.idx[x], a0 = xt.w0[x], a1 = COEF_SCALE - a0;
      const p0 = base + s * 4;
      const p1 = s + 1 < w ? p0 + 4 : p0;
      for (let c = 0; c < 3; c++) row[x * 3 + c] = rgba[p0 + c] * a0 + rgba[p1 + c] * a1;
    }
    rowCache.set(y, row);
    return row;
  };
  const DELTA = 1 << (COEF_BITS * 2 - 1);
  for (let y = 0; y < nh; y++) {
    const s = yt.idx[y], b0 = yt.w0[y], b1 = COEF_SCALE - b0;
    const r0 = hrow(s);
    const r1 = s + 1 < h ? hrow(s + 1) : r0;
    const o = y * nw * 3;
    for (let k = 0; k < nw * 3; k++) {
      const v = (r0[k] * b0 + r1[k] * b1 + DELTA) >> (COEF_BITS * 2);
      out[o + k] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
    for (const k of rowCache.keys()) if (k < s) rowCache.delete(k); // source rows only move forward
  }
  return out;
}

/** SPEC §6.1. `rgba` is width*height*4 bytes (ImageData.data). */
export function letterbox(rgba: ArrayLike<number>, w: number, h: number, imgsz = 640): Letterbox {
  if (!(w > 0 && h > 0) || rgba.length < w * h * 4) throw new Error(`letterbox: bad image ${w}x${h} (${rgba.length} bytes)`);
  const p = letterboxParams(w, h, imgsz);
  const rgb = resizeBilinearRGB(rgba, w, h, p.nw, p.nh);
  const plane = imgsz * imgsz;
  const tensor = new Float32Array(3 * plane).fill(114 / 255);
  for (let y = 0; y < p.nh; y++) {
    const ty = y + p.top;
    if (ty < 0 || ty >= imgsz) continue;
    for (let x = 0; x < p.nw; x++) {
      const tx = x + p.left;
      if (tx < 0 || tx >= imgsz) continue;
      const s = (y * p.nw + x) * 3;
      const t = ty * imgsz + tx;
      tensor[t] = rgb[s] / 255;
      tensor[plane + t] = rgb[s + 1] / 255;
      tensor[2 * plane + t] = rgb[s + 2] / 255;
    }
  }
  return { tensor, imgsz, ...p };
}
