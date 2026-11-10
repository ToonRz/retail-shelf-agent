// Inference pipeline. OWNER: task A (see docs/agent-connection/TASKS.md in
// retail-shelf-detection). Everything here throws "not_implemented" until then;
// load() already does the manifest / sha256 / labels checks before reaching it.

import { AgentError, type Backend, type DetectInput, type DetectOptions, type DetectResult, type Manifest } from "./types";

export interface Session {
  backend: Backend;
  // A1: wrap an onnxruntime-web InferenceSession here.
}

/** A1: create an onnxruntime-web session from the verified model bytes.
 *  Prefer `backend`, fall back from "webgpu" to "wasm". The release is flat: ort's
 *  .wasm/.mjs files sit next to shelf-agent.js, so set ort.env.wasm.wasmPaths = baseUrl. */
export async function createSession(_model: Uint8Array, _baseUrl: string, _backend: Backend): Promise<Session> {
  throw new AgentError("not_implemented", "createSession: task A1");
}

/** A2-A4: letterbox -> run -> decode + NMS -> boxes in original pixels. */
export async function detect(
  _session: Session,
  _image: DetectInput,
  _labels: string[],
  _manifest: Manifest,
  _opts: DetectOptions = {},
): Promise<DetectResult> {
  throw new AgentError("not_implemented", "detect: tasks A2-A4");
}

export async function disposeSession(_session: Session): Promise<void> {}
