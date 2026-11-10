import { createSession, detect, disposeSession } from "./detect";
import { parseManifest, sha256Hex } from "./manifest";
import { type Agent, AgentError, type LoadOptions } from "./types";

export * from "./types";
export { parseManifest, sha256Hex } from "./manifest";

async function fetchOk(url: string): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(url);
  } catch (e) {
    throw new AgentError("fetch_failed", `${url}: ${(e as Error).message}`);
  }
  if (!res.ok) throw new AgentError("fetch_failed", `${url}: HTTP ${res.status}`);
  return res;
}

async function fetchBytes(url: string, onProgress?: LoadOptions["onProgress"]): Promise<Uint8Array> {
  const res = await fetchOk(url);
  const len = Number(res.headers.get("content-length")) || null;
  if (!res.body || !onProgress) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(loaded, len);
  }
  const out = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

/** Load the pinned release from `baseUrl`. Throws AgentError on any mismatch. */
export async function load(opts: LoadOptions): Promise<Agent> {
  const base = opts.baseUrl.endsWith("/") ? opts.baseUrl : `${opts.baseUrl}/`;
  let raw: unknown;
  try {
    raw = await (await fetchOk(`${base}manifest.json`)).json();
  } catch (e) {
    if (e instanceof AgentError) throw e;
    throw new AgentError("manifest_invalid", "manifest.json is not JSON");
  }
  const manifest = parseManifest(raw, opts.expectedVersion);

  const model = await fetchBytes(`${base}${manifest.model.file}`, opts.onProgress);
  const sha = await sha256Hex(model);
  if (sha !== manifest.model.sha256) {
    throw new AgentError("model_sha256_mismatch", `${manifest.model.file} sha256 ${sha} != manifest ${manifest.model.sha256}`);
  }
  const labels = (await (await fetchOk(`${base}${manifest.model.labels}`)).json()) as unknown;
  if (!Array.isArray(labels) || labels.length !== manifest.model.classes || !labels.every((l) => typeof l === "string")) {
    throw new AgentError("labels_mismatch", `${manifest.model.labels} must be ${manifest.model.classes} class names`);
  }

  const session = await createSession(model, base, opts.backend ?? "webgpu");
  return {
    manifest,
    backend: session.backend,
    detect: (image, o) => detect(session, image, labels as string[], manifest, o),
    dispose: () => disposeSession(session),
  };
}
