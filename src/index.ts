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

async function fetchVerifiedModel(
  base: string,
  info: ModelInfo,
  onProgress?: LoadOptions["onProgress"],
): Promise<{ model: Uint8Array; labels: string[] }> {
  const model = await fetchBytes(`${base}${info.file}`, onProgress);
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

  const primary = await fetchVerifiedModel(base, manifest.model, opts.onProgress);
  const shelfLoaded = manifest.shelfModel
    ? await fetchVerifiedModel(
        base,
        manifest.shelfModel,
        opts.onProgress ? (n, t) => opts.onProgress!(primary.model.length + n, t ? primary.model.length + t : null) : undefined,
      )
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
