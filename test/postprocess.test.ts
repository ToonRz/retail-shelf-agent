// A3: decode() with hand-made outputs.
import { describe, expect, it } from "vitest";
import { decode, iou, mergeBoxes, overlapRatio } from "../src/postprocess";
import type { Box } from "../src/types";

const C = 3;
const OPTS = { conf: 0.25, iou: 0.7, maxDet: 300 };
const META = { left: 0, top: 0, gx: 1, gy: 1, width: 640, height: 640 };

/** Build a channel-major [1, 4 + C, N] output from anchors {cx, cy, w, h, scores}. */
function fake(anchors: Array<[number, number, number, number, number[]]>): Float32Array {
  const N = anchors.length;
  const out = new Float32Array((4 + C) * N);
  anchors.forEach(([cx, cy, w, h, s], i) => {
    out[i] = cx; out[N + i] = cy; out[2 * N + i] = w; out[3 * N + i] = h;
    s.forEach((v, c) => { out[(4 + c) * N + i] = v; });
  });
  return out;
}

describe("iou", () => {
  it("is 1 for the same box, 0 for disjoint boxes, 1/3 for half-overlapping squares", () => {
    expect(iou([0, 0, 10, 10], [0, 0, 10, 10])).toBe(1);
    expect(iou([0, 0, 10, 10], [20, 20, 30, 30])).toBe(0);
    expect(iou([0, 0, 10, 10], [5, 0, 15, 10])).toBeCloseTo(1 / 3, 6);
  });
});

describe("decode", () => {
  it("argmax class, conf strictly greater than the threshold, cxcywh -> xyxy, sorted by conf", () => {
    const boxes = decode(fake([
      [100, 100, 20, 40, [0.1, 0.9, 0.2]],
      [300, 300, 10, 10, [0.25, 0.1, 0.1]], // == threshold: dropped (Ultralytics uses >)
      [500, 200, 50, 50, [0.3, 0.2, 0.95]],
    ]), C, OPTS, META, ["a", "b", "c"]);
    expect(boxes.map((b) => [b.cls, b.classId, Number(b.conf.toFixed(2))])).toEqual([["c", 2, 0.95], ["b", 1, 0.9]]);
    expect(boxes[1].xyxy).toEqual([90, 80, 110, 120]);
  });

  it("does not suppress overlapping boxes of different classes", () => {
    const boxes = decode(fake([
      [100, 100, 50, 50, [0.9, 0.1, 0.1]],
      [102, 100, 50, 50, [0.1, 0.8, 0.1]],
    ]), C, OPTS, META);
    expect(boxes.map((b) => b.classId)).toEqual([0, 1]);
  });

  it("suppresses a same-class box with IoU > iou, keeps one with IoU <= iou", () => {
    const boxes = decode(fake([
      [100, 100, 50, 50, [0.9, 0, 0]],
      [102, 100, 50, 50, [0.8, 0, 0]],  // IoU ~0.92 with the first -> suppressed
      [130, 100, 50, 50, [0.7, 0, 0]],  // IoU ~0.25 -> kept
    ]), C, OPTS, META);
    expect(boxes.map((b) => b.conf.toFixed(1))).toEqual(["0.9", "0.7"]);
  });

  it("keeps only max_det boxes", () => {
    const anchors = Array.from({ length: 10 }, (_, i) => [i * 60 + 20, 20, 10, 10, [0.5 + i / 100, 0, 0]] as [number, number, number, number, number[]]);
    const boxes = decode(fake(anchors), C, { ...OPTS, maxDet: 3 }, META);
    expect(boxes).toHaveLength(3);
    expect(boxes[0].conf).toBeCloseTo(0.59, 5);
  });

  it("undoes the letterbox and clips to the original image", () => {
    // 1280x720 letterboxed into 640: gain 0.5, top padding 140
    const meta = { left: 0, top: 140, gx: 0.5, gy: 0.5, width: 1280, height: 720 };
    const [b] = decode(fake([[320, 320, 100, 100, [0.9, 0, 0]]]), C, OPTS, meta);
    expect(b.xyxy).toEqual([540, 260, 740, 460]);
    const [edge] = decode(fake([[5, 150, 30, 40, [0.9, 0, 0]]]), C, OPTS, meta);
    expect(edge.xyxy[0]).toBe(0);  // x1 = (5 - 15) / 0.5 < 0 -> 0
    expect(edge.xyxy[1]).toBe(0);  // y1 = (130 - 140) / 0.5 < 0 -> 0
  });

  it("rejects an output that does not fit C", () => {
    expect(() => decode(new Float32Array(10), C, OPTS, META)).toThrow(/not \(4 \+ 3\)/);
  });
});

describe("mergeBoxes (v0.1.0 110-class + v0.2.1 1-class shelf)", () => {
  const b = (cls: string, classId: number, conf: number, xyxy: [number, number, number, number]): Box => ({ cls, classId, conf, xyxy });

  it("keeps named product boxes, drops overlapping/contained shelf boxes, and keeps unrecognized shelf boxes as 'product'", () => {
    const primary = [
      b("pocky", 73, 0.95, [200, 180, 440, 470]),
    ];
    const shelf = [
      b("product", 0, 0.75, [210, 185, 430, 468]), // overlaps pocky -> dropped
      b("product", 0, 0.60, [10, 290, 90, 510]),   // another item not in 110 classes -> kept with classId 110
      b("product", 0, 0.30, [330, 190, 430, 465]), // sub-box inside pocky (overlapRatio ~1) -> dropped
    ];
    expect(overlapRatio(primary[0].xyxy, shelf[2].xyxy)).toBeGreaterThan(0.9);
    const merged = mergeBoxes(primary, shelf, 110);
    expect(merged).toEqual([
      b("pocky", 73, 0.95, [200, 180, 440, 470]),
      b("product", 110, 0.60, [10, 290, 90, 510]),
    ]);
  });
});
