# retail-shelf-agent

ตัวตรวจจับสินค้าบนชั้นวาง (YOLOv8n) ที่ **รันใน browser** ด้วย onnxruntime-web
ออกเป็น GitHub Release แล้ว [retail-shelf-web](https://github.com/ToonRz/retail-shelf-web) pin เวอร์ชันแล้ว bump มาใช้
(แบบเดียวกับ zcrX ↔ zcrAgentX) มีสองสาย:

| สาย release | model | เว็บ pin ใน | ใช้ที่ |
|---|---|---|---|
| `v0.1.x` | sell-products v3, 110 คลาส (`model.lock.json`) | `agent.version` | หน้าตรวจจับ |
| `v0.2.x` | shelf 1 คลาส `product`, เทรนบน SKU-110K (`model.shelf.lock.json`) | `agent-shelf.version` | หน้าเฝ้าชั้นวาง (แจ้งเตือนของหมดราย slot) |

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
| release แรก `v0.1.0` | ✅ [v0.1.0](https://github.com/ToonRz/retail-shelf-agent/releases/tag/v0.1.0) (2026-11-10) · เว็บ pin แล้ว |
| shelf model 1 คลาส + fixture `shelf-v1` | ✅ (2026-11-12): release `model-shelf-1class-v1` · output `[1,5,8400]` · `test/fixtures/shelf-v1/` ใช้รูป sell-v3 ชุดเดิม (CC BY, เซฟใหม่เป็น JPEG 4:4:4) + 2 เฟรมจากวิดีโอชั้นวางจำลอง เพราะห้ามแจกรูป SKU-110K · release [v0.2.0](https://github.com/ToonRz/retail-shelf-agent/releases/tag/v0.2.0) |
| release `v0.2.0` (shelf model) | ✅ `release.yml` ใช้ `models/shelf` เมื่อ tag เป็น `v0.2.*` |
| shelf model v2 (เทรนครบ 30 epoch บน Colab) + release `v0.2.1` | ✅ (2026-11-12): release `model-shelf-1class-v2` · SKU-110K test tiles mAP50 0.927 (v1: 0.842) · fixture `shelf-v1` สร้างใหม่จาก model นี้ |

## Dev

```bash
npm ci
npm run fetch-model     # models/ ตาม model.lock.json (model-yolov8n-v3, 110 คลาส)
npm run fetch-model:coco-dev   # models/coco-dev/ (model COCO ที่ใช้ทำ fixture coco-dev)
npm run fetch-model:shelf      # models/shelf/ ตาม model.shelf.lock.json (shelf 1 คลาส)
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
- `test/fixtures/shelf-v1/` ↔ `model.shelf.lock.json` (shelf 1 คลาส · รูป sell-v3 แบบ JPEG 4:4:4 คุณภาพ 100 เพราะ jpeg-js กับ libjpeg ถอดรหัสต่างกัน (4:2:0 ต่างได้ ~50 ระดับ, 4:4:4 ไม่เกิน 3) พอจะทำให้คะแนนก้ำกึ่งคลาดเกิน 0.02 · ตัด `IMG_6237` ออกเพราะกล่องของมันไวมาก: onnx ตัวเดียวกันบนพิกเซลที่ jpeg-js ถอดได้ 0.3829 เท่ากับ agent แต่ต่างจาก libjpeg 0.021 · เฟรมวิดีโอย่อเหลือ 640×480)

## License / ที่มา

Model ได้มาจาก Ultralytics YOLOv8 (AGPL-3.0) และ dataset [Sell Products – O2O Minimart](https://universe.roboflow.com/o2o-minimart/sell-products) (CC BY 4.0) ซึ่งรูปใน `test/fixtures/sell-v3/` มาจาก test split ของ dataset นี้
รูป `bus.jpg` ใน fixture มาจาก Ultralytics assets

Shelf model (`v0.2.x`) เทรนบน [SKU-110K](https://github.com/eg4000/SKU110K_CVPR19) ซึ่ง licence ให้ใช้ **เพื่อการศึกษาและไม่ใช่เชิงพาณิชย์เท่านั้น**
และห้ามแจกจ่ายหรือดัดแปลงรูป จึงไม่มีรูป SKU-110K ใน repo นี้ · ถ้าจะใช้ model นี้เชิงพาณิชย์ต้องขออนุญาต Trax ก่อน หรือเทรนใหม่ด้วยรูปของตัวเอง
