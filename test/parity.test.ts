// A4: run model.onnx with onnxruntime-web (wasm, in node) through the same letterbox + decode the
// browser uses, and compare with what Ultralytics predicted from the SAME onnx file (expected.json).
// Criteria: SPEC §6.4. Models come from `npm run fetch-model` (sell-v3) and
// `npm run fetch-model:coco-dev` (coco-dev); CI runs both before the tests.
import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import jpeg from "jpeg-js";
import * as ort from "onnxruntime-web";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { runRGBA, type Session } from "../src/detect";
import { iou } from "../src/postprocess";
import type { Box } from "../src/types";

interface Expected {
  conf: number; iou: number; imgsz: number; model_sha256: string;
  images: Array<{ image: string; width: number; height: number; boxes: Box[] }>;
}

const SETS = [
  { fixtures: "test/fixtures/sell-v3", models: "models" },
  { fixtures: "test/fixtures/coco-dev", models: "models/coco-dev" },
  { fixtures: "test/fixtures/shelf-v1", models: "models/shelf" },
];

/** Greedy one-to-one matching of SPEC §6.4. Returns a list of problems (empty = parity holds). */
export function parityProblems(expected: Box[], got: Box[]): string[] {
  const problems: string[] = [];
  const used = new Set<number>();
  for (const e of expected) {
    let best = -1, bestIou = 0;
    got.forEach((g, j) => {
      if (used.has(j) || g.classId !== e.classId) return;
      const v = iou(e.xyxy, g.xyxy);
      if (v > bestIou) { bestIou = v; best = j; }
    });
    if (best < 0 || bestIou < 0.9) {
      problems.push(`missing ${e.cls} ${e.conf} (best IoU ${bestIou.toFixed(3)})`);
      continue;
    }
    used.add(best);
    const d = Math.abs(got[best].conf - e.conf);
    if (d > 0.02) problems.push(`${e.cls}: conf ${got[best].conf.toFixed(4)} vs ${e.conf} (|Δ| ${d.toFixed(4)})`);
  }
  got.forEach((g, j) => {
    if (!used.has(j) && g.conf >= 0.3) problems.push(`extra ${g.cls} ${g.conf.toFixed(4)}`);
  });
  return problems;
}

describe("parityProblems", () => {
  const b = (classId: number, conf: number, x = 0): Box => ({ cls: String(classId), classId, conf, xyxy: [x, 0, x + 100, 100] });
  it("passes identical sets and flags a wrong class, a conf drift and an extra box", () => {
    expect(parityProblems([b(1, 0.8)], [b(1, 0.79)])).toEqual([]);
    expect(parityProblems([b(1, 0.8)], [b(2, 0.8)])[0]).toMatch(/missing/);
    expect(parityProblems([b(1, 0.8)], [b(1, 0.7)])[0]).toMatch(/conf/);
    expect(parityProblems([], [b(1, 0.31)])[0]).toMatch(/extra/);
    expect(parityProblems([], [b(1, 0.29)])).toEqual([]); // low-conf extras are tolerated
  });
});

for (const set of SETS) {
  describe(`parity on ${set.fixtures}`, () => {
    const exp = JSON.parse(readFileSync(join(set.fixtures, "expected.json"), "utf8")) as Expected;
    const modelPath = join(set.models, "model.onnx");
    let session: Session;
    let labels: string[];

    beforeAll(async () => {
      if (!existsSync(modelPath)) throw new Error(`${modelPath} missing: run npm run fetch-model && npm run fetch-model:coco-dev`);
      const bytes = readFileSync(modelPath);
      expect(createHash("sha256").update(bytes).digest("hex"), "fixtures were made with another onnx").toBe(exp.model_sha256);
      labels = JSON.parse(readFileSync(join(set.models, "labels.json"), "utf8"));
      const s = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
      session = { backend: "wasm", ort: s, inputName: s.inputNames[0], outputName: s.outputNames[0] };
    }, 60_000);
    afterAll(async () => { await session?.ort.release(); });

    it("covers every image in the folder", () => {
      const jpgs = readdirSync(set.fixtures).filter((f) => /\.jpe?g$/i.test(f)).sort();
      expect(exp.images.map((i) => i.image).sort()).toEqual(jpgs);
    });

    for (const img of exp.images) {
      it(`${img.image} (${img.width}x${img.height}, ${img.boxes.length} boxes)`, async () => {
        const raw = jpeg.decode(readFileSync(join(set.fixtures, img.image)), { useTArray: true, formatAsRGBA: true });
        expect([raw.width, raw.height]).toEqual([img.width, img.height]);
        const got = await runRGBA(session, raw, labels, { imgsz: exp.imgsz, conf: exp.conf, iou: exp.iou, max_det: 300 });
        expect(parityProblems(img.boxes, got)).toEqual([]);
      }, 60_000);
    }
  });
}

describe("combined detection (v0.1.0 110-class + v0.2.1 1-class shelf)", () => {
  let primarySession: Session;
  let shelfSession: Session;
  let primaryLabels: string[];
  let shelfLabels: string[];

  beforeAll(async () => {
    primaryLabels = JSON.parse(readFileSync("models/labels.json", "utf8"));
    shelfLabels = JSON.parse(readFileSync("models/shelf/labels.json", "utf8"));
    const p = await ort.InferenceSession.create(readFileSync("models/model.onnx"), { executionProviders: ["wasm"] });
    const s = await ort.InferenceSession.create(readFileSync("models/shelf/model.onnx"), { executionProviders: ["wasm"] });
    primarySession = { backend: "wasm", ort: p, inputName: p.inputNames[0], outputName: p.outputNames[0] };
    shelfSession = { backend: "wasm", ort: s, inputName: s.inputNames[0], outputName: s.outputNames[0] };
  }, 60_000);
  afterAll(async () => {
    await primarySession?.ort.release();
    await shelfSession?.ort.release();
  });

  it("keeps named class from v0.1.0 and adds unrecognized shelf products from v0.2.1 as 'product'", async () => {
    const cfg = { imgsz: 640, conf: 0.25, iou: 0.7, max_det: 300 };
    const shelf = { session: shelfSession, labels: shelfLabels, model: cfg };

    // IMG_5895: v0.1.0 recognizes central 'yen-yen'; v0.2.1 also sees two neighbouring products on the shelf
    const img1 = jpeg.decode(readFileSync("test/fixtures/shelf-v1/IMG_5895_JPG.rf.0774158c96bfb7f9e10ee8acd93a9148.jpg"), { useTArray: true, formatAsRGBA: true });
    const boxes1 = await runRGBA(primarySession, img1, primaryLabels, cfg, {}, shelf);
    expect(boxes1.map((b) => [b.cls, b.classId])).toEqual([
      ["yen-yen", 105],
      ["product", 110],
      ["product", 110],
    ]);

    // 23EB2CED: v0.1.0 returns [], v0.2.1 detects the product -> combined returns ["product", 110]
    const img2 = jpeg.decode(readFileSync("test/fixtures/shelf-v1/23EB2CED-211B-4A4B-950D-55EA5C6A729F_jpg.rf.118d558879077a6fd3bd4159e34bf90d.jpg"), { useTArray: true, formatAsRGBA: true });
    const boxes2 = await runRGBA(primarySession, img2, primaryLabels, cfg, {}, shelf);
    expect(boxes2.map((b) => [b.cls, b.classId])).toEqual([["product", 110]]);
  }, 60_000);
});
