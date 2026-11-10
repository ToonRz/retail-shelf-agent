import { AgentError, CONTRACT, type Manifest } from "./types";

const VERSION_RE = /^v\d+\.\d+\.\d+$/;
const SHA_RE = /^[0-9a-f]{64}$/;

function fail(msg: string): never {
  throw new AgentError("manifest_invalid", `manifest.json: ${msg}`);
}

/** Validate a parsed manifest.json and check it is the release the web pinned. */
export function parseManifest(raw: unknown, expectedVersion: string): Manifest {
  if (typeof raw !== "object" || raw === null) fail("not an object");
  const m = raw as Manifest;
  if (typeof m.version !== "string" || !VERSION_RE.test(m.version)) fail(`bad version ${String(m.version)}`);
  if (m.contract !== CONTRACT) {
    throw new AgentError("contract_mismatch", `release speaks ${String(m.contract)}, this agent speaks ${CONTRACT}`);
  }
  if (m.version !== expectedVersion) {
    throw new AgentError("version_mismatch", `release is ${m.version}, web pinned ${expectedVersion}`);
  }
  if (typeof m.entry !== "string" || !m.entry) fail("missing entry");
  const mo = m.model;
  if (typeof mo !== "object" || mo === null) fail("missing model");
  for (const k of ["file", "labels", "name", "dataset"] as const) {
    if (typeof mo[k] !== "string" || !mo[k]) fail(`model.${k} missing`);
  }
  for (const k of ["classes", "imgsz", "conf", "iou", "max_det"] as const) {
    if (typeof mo[k] !== "number" || !(mo[k] > 0)) fail(`model.${k} must be a positive number`);
  }
  if (!SHA_RE.test(mo.sha256)) fail("model.sha256 must be 64 lowercase hex chars");
  if (typeof m.files !== "object" || m.files === null) fail("missing files");
  for (const name of [m.entry, mo.file, mo.labels]) {
    if (!(name in m.files)) fail(`files does not list ${name}`);
  }
  for (const [name, sha] of Object.entries(m.files)) {
    if (!SHA_RE.test(sha)) fail(`files[${name}] is not a sha256`);
  }
  if (m.files[mo.file] !== mo.sha256) fail("model.sha256 differs from files[model.file]");
  return m;
}

export async function sha256Hex(data: ArrayBuffer | Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", data as BufferSource);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}
