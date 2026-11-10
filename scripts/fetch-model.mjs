#!/usr/bin/env node
// Download the model named in model.lock.json into models/ and verify every sha256.
//   node scripts/fetch-model.mjs
// The lock points at a GitHub release that holds model.onnx, labels.json, model.json
// (written by retail-shelf-detection's src/export_agent_assets.py). Switching to a new
// model = upload it there, then update model.lock.json in a PR.
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";

const lock = JSON.parse(readFileSync("model.lock.json", "utf8"));
mkdirSync("models", { recursive: true });
const sha = (buf) => createHash("sha256").update(buf).digest("hex");

for (const [name, want] of Object.entries(lock.files)) {
  const dst = `models/${name}`;
  if (existsSync(dst) && sha(readFileSync(dst)) === want) {
    console.log(`ok (cached) ${dst}`);
    continue;
  }
  const res = await fetch(lock.source + name);
  if (!res.ok) {
    console.error(`fetch-model: ${lock.source + name}: HTTP ${res.status}`);
    process.exit(1);
  }
  const buf = Buffer.from(await res.arrayBuffer());
  if (sha(buf) !== want) {
    console.error(`fetch-model: ${name} sha256 ${sha(buf)} != lock ${want}`);
    process.exit(1);
  }
  writeFileSync(dst, buf);
  console.log(`ok ${dst} (${buf.length} bytes)`);
}
