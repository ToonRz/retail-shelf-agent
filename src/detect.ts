// Inference pipeline (task A): onnxruntime-web session -> letterbox -> run -> decode.
// load() in index.ts has already checked manifest / sha256 / labels before createSession.
import * as ort from "onnxruntime-web";
import { decode, mergeBoxes } from "./postprocess";
import { type Letterbox, letterbox } from "./preprocess";
import { AgentError, type Backend, type DetectInput, type DetectOptions, type DetectResult, type Manifest } from "./types";

export interface Session {
  backend: Backend;
  ort: ort.InferenceSession;
  inputName: string;
  outputName: string;
}

export interface ShelfRunner {
  session: Session;
  labels: string[];
  model: Pick<Manifest["model"], "imgsz" | "conf" | "iou" | "max_det">;
}

async function hasWebGPU(): Promise<boolean> {
  const gpu = (globalThis.navigator as unknown as { gpu?: { requestAdapter(): Promise<unknown> } } | undefined)?.gpu;
  if (!gpu) return false;
  try {
    return !!(await gpu.requestAdapter());
  } catch {
    return false;
  }
}

/** A1: create a session from the verified model bytes. Prefer `backend`; "webgpu" falls back to
 *  "wasm". The release is flat, so ort's .wasm sits next to shelf-agent.js under `baseUrl`. */
export async function createSession(model: Uint8Array, baseUrl: string, backend: Backend): Promise<Session> {
  if (typeof document !== "undefined") {
    ort.env.wasm.wasmPaths = new URL(baseUrl, document.baseURI).href;
    // GitHub Pages is not cross-origin isolated, so no SharedArrayBuffer threads.
    if (!globalThis.crossOriginIsolated) ort.env.wasm.numThreads = 1;
  }
  const order: Backend[] = backend === "webgpu" ? ["webgpu", "wasm"] : ["wasm"];
  const errors: string[] = [];
  for (const b of order) {
    if (b === "webgpu" && !(await hasWebGPU())) {
      errors.push("webgpu: no adapter");
      continue;
    }
    try {
      const s = await ort.InferenceSession.create(model, { executionProviders: [b], graphOptimizationLevel: "all" });
      return { backend: b, ort: s, inputName: s.inputNames[0], outputName: s.outputNames[0] };
    } catch (e) {
      errors.push(`${b}: ${(e as Error).message}`);
    }
  }
  throw new AgentError("backend_unavailable", `no backend could run the model (${errors.join("; ")})`);
}

/** RGBA pixels of any supported input (browser only for non-ImageData inputs). */
export function toRGBA(image: DetectInput): { data: Uint8ClampedArray; width: number; height: number } {
  if (typeof ImageData !== "undefined" && image instanceof ImageData) return image;
  if ("data" in image && "width" in image) return image as ImageData; // ImageData-like (node tests)
  const el = image as HTMLImageElement;
  const width = el.naturalWidth || image.width, height = el.naturalHeight || image.height;
  if (!(width > 0 && height > 0)) throw new Error("detect: image has no size (not loaded yet?)");
  const canvas = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(width, height) : Object.assign(document.createElement("canvas"), { width, height });
  const ctx = canvas.getContext("2d", { willReadFrequently: true }) as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null;
  if (!ctx) throw new Error("detect: no 2d canvas context");
  ctx.drawImage(image as CanvasImageSource, 0, 0, width, height);
  return { data: ctx.getImageData(0, 0, width, height).data, width, height };
}

async function runLetterbox(
  session: Session,
  lb: Letterbox,
  width: number,
  height: number,
  labels: string[],
  model: Pick<Manifest["model"], "imgsz" | "conf" | "iou" | "max_det">,
  opts: DetectOptions = {},
) {
  const input = new ort.Tensor("float32", lb.tensor, [1, 3, model.imgsz, model.imgsz]);
  const out = await session.ort.run({ [session.inputName]: input });
  const t = out[session.outputName];
  const C = labels.length;
  if (t.dims.length !== 3 || t.dims[1] !== 4 + C) throw new Error(`model output ${JSON.stringify(t.dims)} does not fit ${C} classes`);
  const data = (await t.getData()) as Float32Array;
  t.dispose?.();
  return decode(data, C, { conf: opts.conf ?? model.conf, iou: opts.iou ?? model.iou, maxDet: opts.maxDet ?? model.max_det },
    { left: lb.left, top: lb.top, gx: lb.gx, gy: lb.gy, width, height }, labels);
}

/** Run the model (and optional shelf model) on RGBA pixels. Shared by detect() and the parity test. */
export async function runRGBA(
  session: Session,
  rgba: { data: ArrayLike<number>; width: number; height: number },
  labels: string[],
  model: Pick<Manifest["model"], "imgsz" | "conf" | "iou" | "max_det">,
  opts: DetectOptions = {},
  shelf?: ShelfRunner,
) {
  const lb = letterbox(rgba.data, rgba.width, rgba.height, model.imgsz);
  const primary = await runLetterbox(session, lb, rgba.width, rgba.height, labels, model, opts);
  if (!shelf) return primary;
  const shelfLb = shelf.model.imgsz === model.imgsz ? lb : letterbox(rgba.data, rgba.width, rgba.height, shelf.model.imgsz);
  const shelfBoxes = await runLetterbox(shelf.session, shelfLb, rgba.width, rgba.height, shelf.labels, shelf.model, opts);
  return mergeBoxes(primary, shelfBoxes, labels.length, opts.maxDet ?? model.max_det);
}

export async function detect(
  session: Session,
  image: DetectInput,
  labels: string[],
  manifest: Manifest,
  opts: DetectOptions = {},
  shelf?: ShelfRunner,
): Promise<DetectResult> {
  const t0 = performance.now();
  const rgba = toRGBA(image);
  const boxes = await runRGBA(session, rgba, labels, manifest.model, opts, shelf);
  return {
    version: manifest.version,
    model_sha256: manifest.model.sha256,
    backend: session.backend,
    latency_ms: Math.round((performance.now() - t0) * 10) / 10,
    width: rgba.width,
    height: rgba.height,
    boxes,
  };
}

export async function disposeSession(session: Session): Promise<void> {
  await session.ort.release();
}
