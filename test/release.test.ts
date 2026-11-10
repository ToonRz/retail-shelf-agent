import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
// @ts-expect-error plain .mjs script, no types
import { packageRelease } from "../scripts/package-release.mjs";
import { AgentError, load, parseManifest, sha256Hex } from "../src/index";

const LABELS = ["pepsi", "lays-salt", "mama-pork"];

/** A fake but structurally real release folder (tiny fake model). */
function fakeRelease(version = "v0.1.0") {
  const root = mkdtempSync(join(tmpdir(), "shelf-agent-"));
  const models = join(root, "models"), dist = join(root, "dist"), ort = join(root, "ort");
  for (const d of [models, dist, ort]) mkdirSync(d);
  writeFileSync(join(models, "model.onnx"), "fake-onnx-bytes");
  writeFileSync(join(models, "labels.json"), JSON.stringify(LABELS));
  writeFileSync(join(models, "model.json"), JSON.stringify({ name: "yolov8n", dataset: "test", classes: 3, imgsz: 640 }));
  writeFileSync(join(dist, "shelf-agent.js"), "export {}");
  writeFileSync(join(ort, "ort-wasm-simd-threaded.jsep.wasm"), "wasm");
  writeFileSync(join(ort, "ort-wasm-simd-threaded.jsep.mjs"), "mjs");
  writeFileSync(join(ort, "ort-wasm-simd-threaded.asyncify.wasm"), "other variant, not copied");
  writeFileSync(join(ort, "ort.min.js"), "not copied");
  return { root, ...packageRelease({ version, modelDir: models, dist, ort, out: join(root, "release") }) };
}

/** Serve a release folder through a stubbed global fetch. */
function serve(dir: string, override: Record<string, string> = {}) {
  vi.stubGlobal("fetch", async (url: string) => {
    const name = url.split("/").pop()!;
    const body = override[name] ?? (() => { try { return readFileSync(join(dir, name)); } catch { return null; } })();
    return body === null ? new Response("nope", { status: 404 }) : new Response(body);
  });
}

afterEach(() => vi.unstubAllGlobals());

describe("package-release", () => {
  it("writes a flat release whose manifest passes parseManifest", () => {
    const { dir, manifest } = fakeRelease();
    expect(Object.keys(manifest.files)).toEqual([
      "labels.json", "model.onnx", "ort-wasm-simd-threaded.jsep.mjs", "ort-wasm-simd-threaded.jsep.wasm", "shelf-agent.js",
    ]);
    expect(parseManifest(JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8")), "v0.1.0").model.classes).toBe(3);
  });

  it("SHA256SUMS covers every file including manifest.json, sha256sum format", () => {
    const { dir } = fakeRelease();
    const lines = readFileSync(join(dir, "SHA256SUMS"), "utf8").trim().split("\n");
    expect(lines.map((l) => l.split("  ")[1])).toContain("manifest.json");
    for (const l of lines) {
      const [sum, name] = l.split("  ");
      expect(createHash("sha256").update(readFileSync(join(dir, name))).digest("hex")).toBe(sum);
    }
  });

  it("refuses a bad version and labels that do not match classes", () => {
    expect(() => packageRelease({ version: "0.1" })).toThrow(/vX\.Y\.Z/);
    const { root } = fakeRelease();
    writeFileSync(join(root, "models", "labels.json"), JSON.stringify(["only-one"]));
    expect(() => packageRelease({ version: "v0.1.1", modelDir: join(root, "models"), dist: join(root, "dist"), ort: join(root, "ort"), out: join(root, "r2") }))
      .toThrow(/labels\.json has 1 names/);
  });
});

describe("parseManifest", () => {
  const good = () => JSON.parse(readFileSync(join(fakeRelease().dir, "manifest.json"), "utf8"));
  const code = (f: () => unknown) => { try { f(); } catch (e) { return (e as AgentError).code; } return "ok"; };

  it("rejects the wrong pin, the wrong contract and a tampered model sha", () => {
    expect(code(() => parseManifest(good(), "v0.1.0"))).toBe("ok");
    expect(code(() => parseManifest(good(), "v0.2.0"))).toBe("version_mismatch");
    expect(code(() => parseManifest({ ...good(), contract: "shelf-agent-js/2" }, "v0.1.0"))).toBe("contract_mismatch");
    const m = good();
    m.model.sha256 = "0".repeat(64);
    expect(code(() => parseManifest(m, "v0.1.0"))).toBe("manifest_invalid");
    expect(code(() => parseManifest(null, "v0.1.0"))).toBe("manifest_invalid");
  });
});

describe("load", () => {
  const codeOf = (p: Promise<unknown>) => p.then(() => "ok", (e: AgentError) => e.code);

  it("checks pin, model sha256 and labels before creating a session", async () => {
    const { dir } = fakeRelease();
    serve(dir);
    expect(await codeOf(load({ baseUrl: "/agent/v0.1.0", expectedVersion: "v0.2.0" }))).toBe("version_mismatch");
    serve(dir, { "model.onnx": "tampered" });
    expect(await codeOf(load({ baseUrl: "/agent/v0.1.0/", expectedVersion: "v0.1.0" }))).toBe("model_sha256_mismatch");
    serve(dir, { "labels.json": JSON.stringify(["a"]) });
    expect(await codeOf(load({ baseUrl: "/agent/v0.1.0/", expectedVersion: "v0.1.0" }))).toBe("labels_mismatch");
    serve(dir, { "manifest.json": "<html>" });
    expect(await codeOf(load({ baseUrl: "/x/", expectedVersion: "v0.1.0" }))).toBe("manifest_invalid");
    serve(join(dir, "missing"));
    expect(await codeOf(load({ baseUrl: "/x/", expectedVersion: "v0.1.0" }))).toBe("fetch_failed");
  });

  it("reaches createSession once every check passes, reporting download progress", async () => {
    const { dir } = fakeRelease();
    serve(dir);
    const seen: number[] = [];
    const res = await codeOf(load({ baseUrl: "/agent/v0.1.0/", expectedVersion: "v0.1.0", onProgress: (n) => seen.push(n) }));
    // the fake model bytes pass sha256 but no backend can run them
    expect(res).toBe("backend_unavailable");
    expect(seen.at(-1)).toBe("fake-onnx-bytes".length);
  });

  it("sha256Hex matches node:crypto", async () => {
    expect(await sha256Hex(new TextEncoder().encode("abc"))).toBe(createHash("sha256").update("abc").digest("hex"));
  });
});
