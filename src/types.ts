// Public types of contract shelf-agent-js/1. retail-shelf-web keeps a copy of these
// in src/agent-contract.ts; a change here is a contract change (bump CONTRACT).

export const CONTRACT = "shelf-agent-js/1";

export type Backend = "webgpu" | "wasm";

export interface ModelInfo {
  file: string;        // e.g. "model.onnx", relative to the release folder
  labels: string;      // JSON array of class names in index order
  name: string;        // "yolov8n"
  dataset: string;     // "sell-products v3"
  classes: number;     // 110 for the trained model
  imgsz: number;       // 640 (square, static input)
  conf: number;        // default confidence threshold
  iou: number;         // default NMS IoU
  max_det: number;     // default max boxes per image
  sha256: string;      // of `file`, checked in the browser before the session is created
}

export interface Manifest {
  version: string;                  // vX.Y.Z, equals the release tag
  contract: string;                 // CONTRACT
  entry: string;                    // "shelf-agent.js"
  model: ModelInfo;
  shelfModel?: ModelInfo;           // optional 1-class shelf model bundled in the same release
  files: Record<string, string>;    // every release file except manifest.json and SHA256SUMS -> sha256
}

export interface LoadOptions {
  baseUrl: string;                  // folder that holds manifest.json, e.g. "/agent/v0.1.0/"
  expectedVersion: string;          // from the web's agent.version; a mismatch throws
  backend?: Backend;                // preferred; "webgpu" falls back to "wasm" when unavailable
  onProgress?: (loaded: number, total: number | null) => void; // model download
}

export interface Box {
  cls: string;                      // class name from labels.json
  classId: number;
  conf: number;
  xyxy: [number, number, number, number]; // pixels of the ORIGINAL image
}

export interface DetectOptions {
  conf?: number;
  iou?: number;
  maxDet?: number;
}

export interface DetectResult {
  version: string;
  model_sha256: string;
  backend: Backend;
  latency_ms: number;               // preprocess + run + postprocess
  width: number;                    // original image size
  height: number;
  boxes: Box[];                     // sorted by conf, descending
}

export type DetectInput = ImageBitmap | ImageData | HTMLImageElement | HTMLCanvasElement;

export interface Agent {
  manifest: Manifest;
  backend: Backend;                 // the one actually in use after fallback
  detect(image: DetectInput, opts?: DetectOptions): Promise<DetectResult>;
  dispose(): Promise<void>;
}

export type AgentErrorCode =
  | "fetch_failed"
  | "manifest_invalid"
  | "version_mismatch"
  | "contract_mismatch"
  | "model_sha256_mismatch"
  | "labels_mismatch"
  | "backend_unavailable"
  | "not_implemented";

export class AgentError extends Error {
  constructor(public code: AgentErrorCode, message: string) {
    super(message);
    this.name = "AgentError";
  }
}
