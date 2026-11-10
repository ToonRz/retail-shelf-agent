# retail-shelf-agent

ตัวตรวจจับสินค้าบนชั้นวาง (YOLOv8n, 110 คลาส) ที่ **รันใน browser** ด้วย onnxruntime-web
ออกเป็น GitHub Release แล้ว [retail-shelf-web](https://github.com/ToonRz/retail-shelf-web) pin เวอร์ชันด้วย `agent.version` และ bump มาใช้
(แบบเดียวกับ zcrX ↔ zcrAgentX)

- Spec: [retail-shelf-detection/docs/agent-connection/SPEC.md](https://github.com/ToonRz/retail-shelf-detection/blob/main/docs/agent-connection/SPEC.md)
- งานที่เหลือและเจ้าของงาน: [TASKS.md](https://github.com/ToonRz/retail-shelf-detection/blob/main/docs/agent-connection/TASKS.md)
- Model มาจาก [retail-shelf-detection](https://github.com/ToonRz/retail-shelf-detection) (`src/export_agent_assets.py`)

## API (contract `shelf-agent-js/1`)

```js
import { load } from "/agent/v0.1.0/shelf-agent.js";
const agent = await load({ baseUrl: "/agent/v0.1.0/", expectedVersion: "v0.1.0", backend: "webgpu" });
const r = await agent.detect(imageBitmap);   // { boxes: [{ cls, classId, conf, xyxy }], latency_ms, ... }
```

Types อยู่ใน [`src/types.ts`](src/types.ts) · `load()` ปฏิเสธ (throw `AgentError`) ถ้าเวอร์ชันไม่ตรง pin, contract ไม่ตรง,
sha256 ของ model ไม่ตรง manifest หรือ labels ไม่ครบ

## สถานะ

| ส่วน | สถานะ |
|---|---|
| types, `parseManifest`, `load()` (manifest + sha256 + labels) | ✅ มี test |
| `scripts/package-release.mjs` (โฟลเดอร์ release + `manifest.json` + `SHA256SUMS`) | ✅ มี test |
| `scripts/fetch-model.mjs` + `model.lock.json` | ✅ ชี้ release `model-yolov8n-v3` (model จริง 110 คลาส) |
| `createSession` (WebGPU → WASM) / `detect` (`src/detect.ts`, `src/preprocess.ts`, `src/postprocess.ts`) | ✅ A1–A4 · unit test letterbox เทียบ Python/cv2, decode/NMS, parity บน `sell-v3` + `coco-dev` · ทดสอบใน Chromium แล้วทั้ง WebGPU และ WASM |
| release workflow ตอน push tag (`.github/workflows/release.yml`) | ✅ C1 |
| model จริง (sell-products v3, 110 คลาส) + fixture `sell-v3` | ✅ T2 (2026-11-10): release `model-yolov8n-v3` · output `[1,114,8400]` · `test/fixtures/sell-v3/` 16 รูป (15 + crop) สร้างจาก onnx ไฟล์เดียวกัน |
| release แรก `v0.1.0` | งาน T4 (push tag `v0.1.0`) |

## Dev

```bash
npm ci
npm run fetch-model     # models/ ตาม model.lock.json (model-yolov8n-v3, 110 คลาส)
npm run fetch-model:coco-dev   # models/coco-dev/ (model COCO ที่ใช้ทำ fixture coco-dev)
npm test                # vitest
npm run typecheck
npm run build           # dist/shelf-agent.js
npm run package -- --version v0.0.1   # release/v0.0.1/ (ต้อง build + fetch-model ก่อน)
```

ออก release: `git tag vX.Y.Z && git push origin vX.Y.Z` (หรือ Actions → release → Run workflow แล้วใส่เวอร์ชัน) → `release.yml` รัน typecheck / test / build / package แล้วสร้าง GitHub Release (ไฟล์ flat + `manifest.json` + `SHA256SUMS`)
release มีแค่ ort ชุด `ort-wasm-simd-threaded.jsep.*` (ตัวเดียวรันได้ทั้ง WebGPU และ WASM) รวมทั้ง release ~43 MB

Parity (`test/parity.test.ts`, SPEC §6.4): รัน onnx ด้วย onnxruntime-web (wasm) ผ่าน letterbox + decode ตัวเดียวกับใน browser แล้วเทียบกับ `expected.json` ที่ Ultralytics ทายด้วย **onnx ไฟล์เดียวกัน**
- `test/fixtures/sell-v3/` ↔ `model.lock.json` (model จริง)
- `test/fixtures/coco-dev/` ↔ `test/fixtures/coco-dev/model.lock.json` (COCO dev · รูปที่ต้อง resize จริง)

## License / ที่มา

Model ได้มาจาก Ultralytics YOLOv8 (AGPL-3.0) และ dataset [Sell Products – O2O Minimart](https://universe.roboflow.com/o2o-minimart/sell-products) (CC BY 4.0) ซึ่งรูปใน `test/fixtures/sell-v3/` มาจาก test split ของ dataset นี้
รูป `bus.jpg` ใน fixture มาจาก Ultralytics assets
