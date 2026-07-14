# 声纹识别与实时说话人实名标注 — 设计文档

> 状态：**设计评审中**（未写代码）
> 关联提交：`3981dc08` feat(realtime): 实现会话级说话人分离
> 关联文档：`fix-realtime-transcription-diarization.md`、`speaker-diarization-for-meeting-collections.md`

---

## 1. 背景与问题陈述

### 1.1 当前状态

提交 `3981dc08` 已修复实时转录的说话人分离：录音期间累积完整音频，停止时一次性对整段录音执行 diarization，返回**匿名**标签 `SPEAKER_00` / `SPEAKER_01`。

但仍存在两个未解决的问题：

| 问题 | 说明 |
|------|------|
| **标签永远是匿名的** | 用户看到的是 `SPEAKER_00`，不知道是谁。需手动记忆"0 是张三、1 是李四"。 |
| **跨会话无记忆** | 同一个人在不同录音中可能被分配不同编号。无法关联"上次录音的 SPEAKER_00 = 本次录音的 SPEAKER_01"。 |

### 1.2 目标

引入**持久化声纹库**（voiceprint library）：

1. 用户可**注册**说话人（录入语音样本 → 提取声纹 → 命名保存）
2. 实时转录 `session.finalize` 时，将匿名 `SPEAKER_xx` **匹配为实名**（"张三"/"李四"）
3. 未匹配的说话人保留匿名标签，且可在事后**从已有录音补注册**
4. 声纹跨会话持久，同一人在不同录音中始终映射到同一姓名

---

## 2. 能力调研结论

### 2.1 CrispASR 的声纹能力（`github.com/cloudlnkcn/CrispASR`）

**结论：CrispASR 内部具备完整的声纹注册/匹配/持久化能力，但仅暴露于 CLI/库 API，未暴露于 HTTP 服务端。**

#### 已有能力（CLI / C / Python / C# 库）

| 能力 | 实现位置 | 说明 |
|------|---------|------|
| `.spkr` 磁盘格式 | `src/speaker_db.h`、`src/speaker_db.cpp:25-55` | `magic "SPKR" \| uint32 version \| uint32 dim \| dim×float32(L2归一化)` |
| 声纹注册 | `crispasr_run.cpp:346-391` | `--enroll-speaker NAME --speaker-db DIR --speaker-db-consent` → 写 `<dir>/<name>.spkr` |
| 声纹 1:N 匹配 | `crispasr_run.cpp:875-913`、`speaker_db.cpp:115-146` | 加载 DB → 逐段提取 embedding → 余弦相似度 > 阈值 → 替换为实名 |
| Embedding 模型 | `src/crispasr_speaker_embedder.cpp:45-184` | **TitaNet-Large（192维，默认）** / ECAPA-TDNN（512维，可选） |
| 聚类 | `src/crispasr_speaker_cluster.h` | 单链层次聚类，余弦相似度，默认阈值 0.5 |

#### 关键缺口（HTTP 服务端）

`examples/cli/crispasr_server.cpp` 的完整路由表中，**没有任何声纹相关端点**：

```
POST /inference                         # diarize 字段 → 仅匿名会话标签
POST /v1/audio/transcriptions           # 同上
POST /load                              # 热切换模型
GET  /health | /backends | /v1/models
POST /v1/translate | /v1/audio/speech | /v1/audio/speech-to-speech | /v1/chat/completions
GET/POST /v1/voices                     # TTS 语音克隆，与说话人身份无关
```

`crispasr_server.cpp` 中对 `speaker_db` / `enroll` / `voiceprint` 的匹配数 = **0**。

> **因此：声纹持久化能力存在，但被锁在 CLI/库里，HTTP 客户端（包括 kyserai）无法触达。**

### 2.2 Kyserai 当前状态

- `CrispASRServer`（`src/cpp/server/backends/crispasr_server.cpp`）已托管下载 **TitaNet-Large** 与 **pyannote 3.0 分割** 模型（`ensure_titanet_model()` :364、`ensure_pyannote_segment_model()` :351）。
- diarize 参数链路已打通：`session.update` → `transcribe_wav(is_session_final=true)` → multipart 转发（`realtime_session.cpp:556-583`）。
- 代码库中 `voiceprint` / `声纹` / `speaker_profile` / `enroll_voice` 关键词 **零命中** —— 无任何持久化声纹存储。
- `IVoiceManagementServer` / `/v1/audio/voices` 是 **TTS 语音克隆**（参考音频管理），与说话人身份识别无关。

### 2.3 架构约束（决定方案选择）

AGENTS.md **不变量 #3**：后端必须作为子进程运行，**不能** in-process 链接 CrispASR 的 C 库直接调用 `speaker_db_*` API。Kyserai 与 CrispASR 之间唯一的通道是 **HTTP**。

因此，要复用 CrispASR 已验证的 `.spkr` 格式、TitaNet embedding、余弦匹配，就必须**在 CrispASR HTTP 服务端新增声纹端点**，kyserai 代理之。

---

## 3. 推荐方案：A — 扩展 CrispASR HTTP + kyserai 代理

### 3.1 方案对比与选择理由

| 维度 | A. 扩展 CrispASR HTTP（推荐） | B. kyserai 本地声纹库 | C. 实时流式声纹 |
|------|:---:|:---:|:---:|
| 复用 CrispASR `.spkr`/TitaNet/匹配 | ✅ 全复用 | ❌ 需自建 embedding 管线 | ✅ 全复用 |
| 符合不变量 #3（子进程 HTTP） | ✅ | ⚠️ 需独立加载 TitaNet | ✅ |
| 实现实名标注改动量 | 中 | 中高 | 高 |
| 支持实时（非仅 finalize）标注 | ❌ 仅 finalize | ✅ 可实时 | ✅ |
| 需上游 CrispASR PR | ✅ 是 | ❌ 否 | ✅ 是 |

**选择 A 的理由：**
1. **最大化复用**：CrispASR 的 `.spkr` 格式、TitaNet embedding 提取、余弦 1:N 匹配已生产可用，重写无价值且易错。
2. **符合架构**：kyserai 始终是 HTTP 透传代理，新增端点沿用现有 `forward_*` 模式，一致性最高。
3. **改动收敛**：声纹逻辑全部在 CrispASR，kyserai 只加薄代理 + 声纹目录管理 + 实时会话的一处匹配点。
4. **实时性取舍可接受**：当前架构已是"录音中纯文本、停止后 diarized"（`fix-realtime-transcription-diarization.md` 的 UX 取舍），实名匹配发生在 finalize 阶段与现有流程一致。方案 C（流式实时实名）可作为后续增量。

### 3.2 整体数据流

```
┌─────────────┐      注册（录音/上传音频）
│  前端 UI     │──────────────────────────────┐
│ 说话人管理   │                               ▼
└──────┬──────┘                     ┌─────────────────┐
       │ POST /v1/audio/speakers     │   kyseraid      │
       │ (multipart: name+audio)     │   (代理层)       │
       │                             │  透传至 CrispASR │
       │                             └────────┬────────┘
       │                                      │ POST /v1/speakers/enroll
       │                                      ▼
       │                             ┌─────────────────┐
       │                             │   CrispASR      │
       │                             │ 提取TitaNet声纹  │
       │                             │ 存 .spkr 文件    │
       │                             └─────────────────┘
       │
       │  实时转录 session.finalize
       │  (携带 speaker_db 引用)
       ▼
┌─────────────┐   整段录音 + speaker_db    ┌─────────────────┐
│ WebSocket   │ ─────────────────────────► │   CrispASR      │
│ /realtime   │                            │ diarize + 匹配   │
└─────────────┘ ◄────────────────────────  │ 返回实名段落     │
   session_final                           └─────────────────┘
   (segments[].speaker = "张三")
```

---

## 4. 上游变更：CrispASR HTTP 声纹端点

> 本节描述需向上游 `cloudlnkcn/CrispASR` 提交的 PR 内容。kyserai 依赖这些端点。

### 4.1 新增端点

所有端点接受标准 multipart/form-data 或 JSON。声纹目录由启动参数 `--speaker-db <dir>` 指定（复用现有 CLI flag）。

| 方法 | 路径 | 用途 | 对应 CLI 能力 |
|------|------|------|--------------|
| `POST` | `/v1/speakers/enroll` | 注册/更新一个说话人声纹 | `--enroll-speaker` |
| `GET` | `/v1/speakers` | 列出已注册说话人 | 读 `<dir>/*.spkr` |
| `DELETE` | `/v1/speakers/{name}` | 删除一个说话人 | 删 `<dir>/<name>.spkr` |
| `GET` | `/v1/speakers/{name}` | 查询单个说话人元信息 | 读单个 `.spkr` |
| `POST` | `/v1/speakers/match` | 对一段音频做 1:N 识别（返回最佳匹配 name+score） | `speaker_db` 匹配逻辑 |

### 4.2 端点详情

#### POST `/v1/speakers/enroll`

**请求**（multipart/form-data）：
| 字段 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `name` | string | ✅ | 说话人姓名（文件名安全字符） |
| `file` | binary | ✅ | 音频样本（wav/mp3/flac，建议 ≥5s 干净人声） |
| `embedder` | string | — | embedding 模型路径；缺省用服务端默认 TitaNet |
| `consent` | string("true") | ✅ | 显式同意门控（对齐 CLI `--speaker-db-consent`） |

**响应** `200 OK`：
```json
{
  "name": "张三",
  "embedding_dim": 192,
  "duration_sec": 8.32,
  "stored": true,
  "path": "speakers/张三.spkr"
}
```

**错误**：`409` 同名已存在（除非加 `overwrite=true`）；`400` 音频太短/无人声；`403` 缺 `consent`。

#### GET `/v1/speakers`

**响应**：
```json
{
  "speakers": [
    {"name": "张三", "embedding_dim": 192, "created_at": "2026-07-14T..."},
    {"name": "李四", "embedding_dim": 192, "created_at": "2026-07-13T..."}
  ],
  "count": 2,
  "embedder": "titanet-large"
}
```

#### POST `/v1/speakers/match`

**请求**（multipart）：`file`(binary) + `threshold`(float, 默认 0.5) + `top_k`(int, 默认 1)

**响应**：
```json
{
  "matches": [
    {"name": "张三", "score": 0.87},
    {"name": "李四", "score": 0.41}
  ],
  "best": {"name": "张三", "score": 0.87},
  "matched": true
}
```

`matched=false` 当 best score < threshold（未注册的陌生人）。

#### diarization 端点增强（`/v1/audio/transcriptions`）

新增**可选**字段（与现有 `diarize_*` 平级）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `speaker_db` | string | 声纹库目录路径或库名。存在时，diarization 的 `segments[].speaker` 输出**实名**而非 `SPEAKER_xx`；未匹配段保留 `SPEAKER_xx`。 |
| `speaker_threshold` | float | 匹配阈值，默认 0.5 |

> 这是对 CrispASR 现有 `speaker_db` 匹配逻辑（`crispasr_run.cpp:887-913`）的 **HTTP 暴露**，不改变算法。

### 4.3 启动参数

CrispASR 服务端新增可选启动参数：

```
--speaker-db <dir>          # 声纹库根目录（持久化 .spkr 文件）
--speaker-db-consent        # 全局同意标志（允许 HTTP enroll）
```

当 `--speaker-db` 未指定时，所有 `/v1/speakers/*` 端点返回 `503`（未配置）。

---

## 5. Kyserai 后端变更

### 5.1 声纹库目录管理

复用 kyserai cache 目录结构（与 `voices/` 平行）：

```
<cache_dir>/
├── voices/          # 已有：TTS 语音克隆
└── speakers/        # 新增：说话人声纹库
    ├── 张三.spkr
    └── 李四.spkr
```

**`CrispASRServer` 变更**（`src/cpp/server/backends/crispasr_server.cpp`）：

- `load()` 中新增：`speaker_db_dir_ = cache_base / "speakers"; fs::create_directories(speaker_db_dir_);`（参照 :448-456 voices 的写法）
- `build_server_args()` 新增：当 `speaker_db_dir_` 存在时传 `--speaker-db <dir> --speaker-db-consent` 给 crispasr 子进程（参照 :152-157 sherpa-segment-model 的传参方式）

### 5.2 新增能力接口

`src/cpp/include/kyserai/server_capabilities.h` 新增：

```cpp
// 说话人声纹管理能力（说话人识别/注册，区别于 TTS 的 IVoiceManagementServer）
class ISpeakerManagementServer : public virtual ICapability {
public:
    virtual ~ISpeakerManagementServer() = default;
    virtual json list_speakers() = 0;
    virtual json enroll_speaker(const std::string& name,
                                const std::string& audio_data,
                                const std::string& filename,
                                const std::string& embedder,
                                bool overwrite) = 0;
    virtual json delete_speaker(const std::string& name) = 0;
    virtual json match_speaker(const std::string& audio_data,
                               const std::string& filename,
                               double threshold, int top_k) = 0;
};
```

`CrispASRServer` 继承此接口，实现为对 `http://127.0.0.1:<port>/v1/speakers/*` 的 HTTP 代理（参照 `forward_transcription_request()` :687、`list_voices()` :954 的模式）。

### 5.3 新增 REST 端点（四前缀注册）

`src/cpp/server/server.cpp` 注册（参照 :657-666 voices 的注册模式）：

```cpp
register_get("audio/speakers",   handle_audio_speakers_list);
register_post("audio/speakers",  handle_audio_speakers_enroll);
register_delete("audio/speakers", handle_audio_speakers_delete);
register_post("audio/speakers/match", handle_audio_speakers_match);
```

> **不变量 #1（四前缀）**：`register_*` 已封装 `/api/v0/`、`/api/v1/`、`/v0/`、`/v1/` 四前缀注册，沿用即可。

Handler 实现（参照 :3569-3643 voices handler）：
- `handle_audio_speakers_list`：调 `router_->list_speakers(model)`
- `handle_audio_speakers_enroll`：解析 multipart（`name`, `file`, `embedder?`, `overwrite?`），调 `router_->enroll_speaker(...)`
- `handle_audio_speakers_delete`：从路径提取 `{name}`，调 `router_->delete_speaker(name)`
- `handle_audio_speakers_match`：解析 multipart（`file`, `threshold?`, `top_k?`），调 `router_->match_speaker(...)`

### 5.4 Router 层

`src/cpp/server/router.cpp` 新增（参照现有 `list_voices`/`upload_voice` 的路由方式）：
- `list_speakers(model)` / `enroll_speaker(...)` / `delete_speaker(...)` / `match_speaker(...)`
- 路由到当前加载的 audio 后端（`CrispASRServer`），通过 `supports_capability<ISpeakerManagementServer>` 检查能力。

### 5.5 实时会话集成（实名匹配的关键点）

**核心改动在 `realtime_session.cpp:556-583` 的 `transcribe_wav(is_session_final=true)`**：

当前：
```cpp
if (session->diarize && is_session_final) {
    request["diarize"] = true;
    request["response_format"] = "diarized_json";
    // ... diarize_method, embedder, cluster_threshold, max_speakers
}
```

新增（在 diarize 块内）：
```cpp
// 声纹库实名匹配：当用户启用了 speaker_db，CrispASR 将 SPEAKER_xx
// 替换为实名；未匹配的保留匿名标签。
if (!session->speaker_db.empty()) {
    request["speaker_db"] = session->speaker_db;
    request["speaker_threshold"] = session->speaker_threshold;
}
```

**`RealtimeSession` 结构体新增字段**（`realtime_session.h:43-52` diarize 块附近）：

```cpp
// 声纹库实名匹配（speaker identification）。当非空时，session-final
// diarization 会将匿名 SPEAKER_xx 标签替换为已注册说话人的姓名。
std::string speaker_db;          // 声纹库名或路径（"default" 指向 <cache>/speakers）
double speaker_threshold = 0.5;  // 余弦匹配阈值
bool enable_speaker_id = false;  // 总开关
```

**`update_session()` 解析新字段**（`realtime_session.cpp:179-198` diarize 解析块之后）：

```cpp
if (config.contains("speaker_db") && config["speaker_db"].is_string()) {
    session->speaker_db = config["speaker_db"].get<std::string>();
    session->enable_speaker_id = !session->speaker_db.empty();
}
if (config.contains("speaker_threshold") && config["speaker_threshold"].is_number()) {
    session->speaker_threshold = config["speaker_threshold"].get<double>();
}
```

> **注意**：实名匹配仅在 `session.finalize` 的整段录音 diarization 时生效。逐块实时转录仍为纯文本（与 `fix-realtime-transcription-diarization.md` 的 UX 取舍一致）。若需实时实名，见第 9 节后续演进。

### 5.6 声纹目录解析

`speaker_db` 字段值解析（在 `transcribe_wav` 或 `CrispASRServer::forward_*` 中）：
- `"default"` 或空 → 解析为 `<cache_dir>/speakers`（kyserai 托管目录）
- 绝对路径 → 原样传递（高级用户/会议集合场景）
- 其他 → 当作相对 `<cache_dir>/speakers/<name>` 的子库名

---

## 6. 前端协议变更

### 6.1 WebSocket `session.update` 新增字段

```typescript
// 前端发送
{
  type: "session.update",
  session: {
    model: "...",
    diarize: true,
    // 新增：
    speaker_db: "default",        // 启用声纹实名匹配
    speaker_threshold: 0.5        // 可选
  }
}
```

### 6.2 `session_final` 消息不变

`segments[].speaker` 字段现在可能是实名（"张三"）或匿名（"SPEAKER_00"，未匹配）。前端渲染逻辑无需改动 —— 它已按 `speaker` 字段着色/分组（见 `fix-realtime-transcription-diarization.md`）。

### 6.3 新增 REST API 调用（`window.api` 契约）

`src/app/src/renderer/tauriShim.ts` 与 C++ 注入的 mock（`server.cpp`）新增方法：

```typescript
interface SpeakerApi {
  listSpeakers(): Promise<{ speakers: SpeakerInfo[]; count: number }>;
  enrollSpeaker(name: string, audio: Blob, opts?: { embedder?: string; overwrite?: boolean }): Promise<EnrollResult>;
  deleteSpeaker(name: string): Promise<void>;
  matchSpeaker(audio: Blob, opts?: { threshold?: number; top_k?: number }): Promise<MatchResult>;
}

interface SpeakerInfo {
  name: string;
  embedding_dim: number;
  created_at: string;
}
```

### 6.4 两种注册形态（用户已确认都要）

#### 形态 A：录音注册

在设置/说话人管理面板：
1. 用户输入姓名 → 点击"录制声纹"
2. 申请麦克风权限 → 录制 5-15 秒（前端用现有 `StreamingAudioBuffer` 的麦克风采集，或简化为 MediaRecorder）
3. 停止 → 调 `enrollSpeaker(name, blob)`
4. 展示结果（embedding_dim、时长、成功/失败）

#### 形态 B：从已有音频注册

在历史转录记录或文件上传转录结果中：
1. 每段 `segment`（带 `speaker` 标签）旁加"注册为说话人"按钮
2. 点击 → 弹出姓名输入 → 该段音频的源（历史录音文件 / 上传的文件对应区间）裁剪后调 `enrollSpeaker`
3. 对于实时转录：`session_recording` 在 finalize 后可按 segment 的 `start/end` 裁剪出该说话人的音频片段用于注册

> **UI 落地说明**：按 AGENTS.md，UI 改动由核心维护者处理。本设计仅定义 API 契约与交互流程，不包含组件实现。

---

## 7. 数据结构与文件格式

### 7.1 `.spkr` 文件格式（CrispASR 定义，kyserai 透传不解析）

```
偏移  大小        内容
0     4 bytes    magic "SPKR"
4     4 bytes    uint32 version (LE)
8     4 bytes    uint32 dim (LE，如 192)
12    dim×4 bytes float32[] L2-归一化 embedding (LE)
```

kyserai **不读写** `.spkr` 内容 —— 仅作为文件由 CrispASR 管理。kyserai 只负责目录的创建与生命周期。

### 7.2 声纹库目录布局

```
<cache_dir>/speakers/
├── 张三.spkr
├── 李四.spkr
└── meeting-q3/              # 子库示例（speaker_db="meeting-q3"）
    ├── 张三.spkr
    └── 王五.spkr
```

### 7.3 实时转录 segment 结构（前后端一致）

```json
{
  "start": 12.34,
  "end": 15.67,
  "text": "你好，今天天气不错。",
  "speaker": "张三"          // 实名（已匹配）或 "SPEAKER_00"（未匹配）
}
```

---

## 8. 改动清单（按文件）

### 上游 CrispASR（独立 PR）
| 文件 | 改动 |
|------|------|
| `examples/cli/crispasr_server.cpp` | 新增 `/v1/speakers/*` 路由；`/v1/audio/transcriptions` 解析 `speaker_db`/`speaker_threshold` |
| `src/speaker_db.h` / `.cpp` | 暴露给 server 用的 enroll/list/delete/match HTTP 友好封装 |
| 启动参数 | `--speaker-db`、`--speaker-db-consent` 在 server 模式下生效 |

### Kyserai C++
| 文件 | 改动 |
|------|------|
| `src/cpp/include/kyserai/server_capabilities.h` | 新增 `ISpeakerManagementServer` 接口（:107 后） |
| `src/cpp/include/kyserai/backends/crispasr_server.h` | `CrispASRServer` 继承 `ISpeakerManagementServer`；声明 `speaker_db_dir_`、4 个方法 |
| `src/cpp/server/backends/crispasr_server.cpp` | `load()` 建 `speakers/` 目录；`build_server_args()` 传 `--speaker-db`；实现 4 个代理方法（参照 `forward_transcription_request` :687） |
| `src/cpp/server/router.cpp` + `.h` | 新增 `list_speakers`/`enroll_speaker`/`delete_speaker`/`match_speaker` 路由方法 |
| `src/cpp/server/server.cpp` | 注册 4 个端点（:666 后）；4 个 handler（参照 :3569-3680）；mock `window.api.speakers.*`（参照现有 voices 注入） |
| `src/cpp/include/kyserai/realtime_session.h` | `RealtimeSession` 新增 `speaker_db`/`speaker_threshold`/`enable_speaker_id`（:52 后） |
| `src/cpp/server/realtime_session.cpp` | `update_session()` 解析新字段（:198 后）；`transcribe_wav` session-final 块加 `speaker_db` 转发（:582 后） |

### Kyserai 前端（核心维护者实施）
| 文件 | 改动 |
|------|------|
| `src/app/src/renderer/tauriShim.ts` | 新增 `speakers` API 契约 |
| `src/app/src/renderer/utils/websocketClient.ts` | `session.update` 传 `speaker_db`（:294 finalize 上方） |
| 说话人管理面板（新组件） | 录音注册 + 已有音频注册 UI |
| `TranscriptionPanel.tsx` | segment 旁"注册为说话人"入口；`session.update` 启用 speaker_db |
| i18n `en.json` / `zh-CN.json` | 新增说话人管理文案 |

---

## 9. 不变量检查

| 不变量 | 遵守情况 |
|--------|---------|
| #1 四前缀注册 | ✅ `register_*` 自动注册四前缀 |
| #2 WrappedServer 契约 | ✅ 新接口是 `ICapability` 可选能力，不影响核心虚拟方法 |
| #3 子进程模型 | ✅ 声纹逻辑在 CrispASR 子进程，kyserai 仅 HTTP 代理 |
| #4 Recipe 完整性 | ✅ 不涉及 server_models.json |
| #5 跨平台 | ✅ 路径用 `fs::path`；无平台特定代码 |
| #6 无硬编码路径 | ✅ 声纹库在 `<cache_dir>/speakers`，用 `path_from_utf8(get_cache_dir())` |
| #7 线程安全 | ✅ handler 无状态；`speaker_db_dir_` 在 `load()` 时设定后只读；实时会话字段在 session 锁保护下设置 |
| #8 Ollama 兼容 | ✅ 不涉及 `/api/*` |
| #9 API key 透传 | ✅ 新端点经 `register_*` 注册，自动套用鉴权中间件 |
| #10 客户端本地状态 | ✅ 说话人库是服务端共享资源（声纹需跨客户端一致），**不是**每客户端设置 |
| #11 web-app 依赖 | ✅ 不涉及 Debian 打包依赖 |
| #12 桌面按需 | ✅ 不影响 kyseraid 生命周期 |

---

## 10. 实施阶段（建议）

### 阶段 0：上游对齐（前置）
- 向 CrispASR 提 PR：新增 `/v1/speakers/*` 端点 + `--speaker-db` server 支持 + `/v1/audio/transcriptions` 的 `speaker_db` 字段
- kyserai 锁定依赖该 CrispASR 版本（更新 `backend_versions.json`）

### 阶段 1：后端代理（无 UI，可用 curl 验证）
- `ISpeakerManagementServer` 接口 + `CrispASRServer` 实现
- 4 个 REST 端点 + router 方法
- 集成测试：`test/server_endpoints.py` 新增 speaker CRUD 用例

### 阶段 2：实时会话实名匹配
- `RealtimeSession` 新增字段 + `transcribe_wav` 转发 `speaker_db`
- WebSocket `session.update` 解析
- 集成测试：`test/server_whisper.py` 新增"注册说话人 → 实时转录 → 验证实名 segment"用例

### 阶段 3：前端（核心维护者）
- `tauriShim` 契约 + 说话人管理面板 + 录音注册 + 已有音频注册

### 阶段 4：实时实名（可选增量，方案 C 演进）
- 若 finalize 后才知实名不够及时，可在 CrispASR 新增流式 embedding 推送
- kyserai 在 `transcribe_interim` 时叠加实时匹配
- 优先级低：当前 finalize UX 已可接受

---

## 11. 风险与未决事项

| 项 | 说明 | 缓解 |
|----|------|------|
| **上游 PR 接受度** | CrispASR 是独立项目，新增 HTTP 端点需其维护者接受 | 设计对齐其现有 CLI 能力，改动最小化；附使用场景说明 |
| **TitaNet 首次注册耗时** | 首次 enroll 需加载 TitaNet（~46MB），可能数秒 | 前端展示 loading；CrispASR 可在 `--speaker-db` 启用时预加载 embedder |
| **声纹质量** | 短样本/噪声导致误匹配 | enroll 建议最短时长（≥5s）；返回 score 供前端展示置信度；阈值可调 |
| **隐私/合规** | 声纹属生物特征 | 复用 CrispASR 的 `consent` 门控；enroll 需显式 `consent=true`；删除即清除 |
| **多语言姓名** | 中文姓名作文件名 | CrispASR `.spkr` 以 name 为文件名；需校验文件名安全（kyserai 侧 sanitize） |
| **匿名→实名映射一致性** | 同一匿名 ID 的多段应映射到同一实名 | CrispASR 内部按聚类后的 speaker ID 统一匹配，非逐段独立匹配 |

---

## 12. 开放问题（待评审确认）

1. **`speaker_db` 的粒度**：全局单一库（`<cache>/speakers`）是否足够？还是需要支持多场景子库（`meeting-q3/`、`podcast/`）？多子库会增加 UI 复杂度。
2. **自动注册策略**：finalize 后出现未匹配的 `SPEAKER_02`，是否自动提示"检测到新说话人，是否注册？"还是完全手动？
3. **阈值默认值**：0.5（CrispASR 默认）是否合适？过高漏认、过误认他人。可能需按 TitaNet 特性调优。
4. **是否需要 embedding 导出/导入**：批量迁移声纹库（如换设备）时，是否需要打包导出？当前设计文件即 `.spkr`，直接拷贝目录即可，但需文档化。
