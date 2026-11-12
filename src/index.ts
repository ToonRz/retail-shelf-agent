import { createSession, detect, disposeSession, type ShelfRunner } from "./detect";
import { parseManifest, sha256Hex } from "./manifest";
import { type Agent, AgentError, type LoadOptions, type ModelInfo } from "./types";

export * from "./types";
export { parseManifest, sha256Hex } from "./manifest";
export { mergeBoxes, overlapRatio } from "./postprocess";

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

/** Read a response body, calling onChunk(bytes) for every chunk received. */
async function readBytes(res: Response, onChunk?: (bytes: number) => void): Promise<Uint8Array> {
  if (!res.body || !onChunk) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onChunk(value.length);
  }
  const out = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

async function verifyModel(base: string, info: ModelInfo, model: Uint8Array): Promise<{ model: Uint8Array; labels: string[] }> {
  const sha = await sha256Hex(model);
  if (sha !== info.sha256) {
    throw new AgentError("model_sha256_mismatch", `${info.file} sha256 ${sha} != manifest ${info.sha256}`);
  }
  const labels = (await (await fetchOk(`${base}${info.labels}`)).json()) as unknown;
  if (!Array.isArray(labels) || labels.length !== info.classes || !labels.every((l) => typeof l === "string")) {
    throw new AgentError("labels_mismatch", `${info.labels} must be ${info.classes} class names`);
  }
  return { model, labels: labels as string[] };
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

  // Open every model download first so onProgress reports one total (sum of the Content-Lengths)
  // that never changes mid-load, then read them in order (primary first, as before).
  const infos = manifest.shelfModel ? [manifest.model, manifest.shelfModel] : [manifest.model];
  const responses = await Promise.all(infos.map((info) => fetchOk(`${base}${info.file}`)));
  const lens = responses.map((r) => Number(r.headers.get("content-length")) || 0);
  const total = lens.every((n) => n > 0) ? lens.reduce((a, b) => a + b, 0) : null;
  let loaded = 0;
  const onChunk = opts.onProgress
    ? (bytes: number) => {
        loaded += bytes;
        opts.onProgress!(loaded, total);
      }
    : undefined;
  const primary = await verifyModel(base, manifest.model, await readBytes(responses[0], onChunk));
  const shelfLoaded = manifest.shelfModel
    ? await verifyModel(base, manifest.shelfModel, await readBytes(responses[1], onChunk))
    : null;

  const session = await createSession(primary.model, base, opts.backend ?? "webgpu");
  let shelf: ShelfRunner | undefined;
  if (shelfLoaded && manifest.shelfModel) {
    try {
      const shelfSession = await createSession(shelfLoaded.model, base, session.backend);
      shelf = { session: shelfSession, labels: shelfLoaded.labels, model: manifest.shelfModel };
    } catch (e) {
      await disposeSession(session);
      throw e;
    }
  }
  return {
    manifest,
    backend: session.backend,
    detect: (image, o) => detect(session, image, primary.labels, manifest, o, shelf),
    dispose: async () => {
      await disposeSession(session);
      if (shelf) await disposeSession(shelf.session);
    },
  };
}
