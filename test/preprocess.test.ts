// A2: letterbox geometry and pixels vs Ultralytics 8.4.208 / cv2 (values in fixtures/letterbox-ref.json
// were produced by LetterBox.get_params and cv2.resize(INTER_LINEAR) in Python).
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { letterbox, letterboxParams, resizeBilinearRGB, roundHalfEven } from "../src/preprocess";

const ref = JSON.parse(readFileSync(new URL("./fixtures/letterbox-ref.json", import.meta.url), "utf8")) as {
  params: Array<{ w: number; h: number; r: number; nw: number; nh: number; left: number; top: number }>;
  resize: Array<{ w: number; h: number; nw: number; nh: number; rgb: number[]; out: number[] }>;
};

const rgbToRgba = (rgb: number[]) => {
  const a = new Uint8ClampedArray((rgb.length / 3) * 4);
  for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) a.set([rgb[i], rgb[i + 1], rgb[i + 2], 255], j);
  return a;
};

describe("roundHalfEven = Python round()", () => {
  it("rounds ties to even, unlike Math.round", () => {
    expect([0.5, 1.5, 2.5, 3.5, -0.5, -1.5, 79.9, 2.4999].map(roundHalfEven).map((v) => v + 0)).toEqual([0, 2, 2, 4, 0, -2, 80, 2]);
    expect(Math.round(2.5)).toBe(3); // the trap this function avoids
  });
});

describe("letterboxParams", () => {
  it.each(ref.params)("$w x $h matches Python", ({ w, h, r, nw, nh, left, top }) => {
    const p = letterboxParams(w, h, 640);
    expect(p.r).toBeCloseTo(r, 12);
    expect([p.nw, p.nh, p.left, p.top]).toEqual([nw, nh, left, top]);
  });
});

describe("resizeBilinearRGB = cv2.resize INTER_LINEAR", () => {
  it.each(ref.resize)("$w x $h -> $nw x $nh", ({ w, h, nw, nh, rgb, out }) => {
    const got = Array.from(resizeBilinearRGB(rgbToRgba(rgb), w, h, nw, nh));
    const maxDiff = Math.max(...got.map((v, i) => Math.abs(v - out[i])));
    expect(maxDiff).toBeLessThanOrEqual(1); // OpenCV's SIMD path can round 1 level differently
  });
});

describe("letterbox", () => {
  it("640x640 is copied without resizing, RGB / 255 channel-major", () => {
    const rgba = new Uint8ClampedArray(640 * 640 * 4);
    rgba.set([255, 128, 0, 255], (10 * 640 + 20) * 4);
    const lb = letterbox(rgba, 640, 640);
    expect([lb.r, lb.nw, lb.nh, lb.left, lb.top]).toEqual([1, 640, 640, 0, 0]);
    const t = 10 * 640 + 20, plane = 640 * 640;
    expect(lb.tensor[t]).toBe(1);
    expect(lb.tensor[plane + t]).toBeCloseTo(128 / 255, 6); // float32
    expect(lb.tensor[2 * plane + t]).toBe(0);
  });

  it("pads with 114 around a non-square image", () => {
    const lb = letterbox(new Uint8ClampedArray(1280 * 720 * 4).fill(255), 1280, 720);
    expect([lb.nw, lb.nh, lb.left, lb.top]).toEqual([640, 360, 0, 140]);
    expect(lb.tensor[0]).toBeCloseTo(114 / 255, 6);              // top padding
    expect(lb.tensor[140 * 640]).toBe(1);                         // first image row
    expect(lb.tensor[(140 + 360) * 640]).toBeCloseTo(114 / 255, 6); // bottom padding
    expect(lb.gx).toBe(0.5);
    expect(lb.gy).toBe(0.5);
  });

  it("refuses a buffer that is too small", () => {
    expect(() => letterbox(new Uint8ClampedArray(10), 10, 10)).toThrow(/bad image/);
  });
});
